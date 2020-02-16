import sys
import time
from datetime import timedelta
from torch.autograd import Variable

from DeepHyperX.batch import DEEPHYPERX_PATH_LINUX
from DeepHyperX.pruning.compressor import *
from DeepHyperX.utils import print_memory_metrics


def iter_prune(args, train_loader, val_loader, the_model=None, stop_percent=None, df_column_entry_dict=None, **kwargs):
    topk = tuple(sorted(kwargs['topk'])) # sort in ascending order

    epoch = 1
    best_loss = sys.maxsize

    assert the_model is not None
    model = the_model # pass actual model from parameters

    import DeepHyperX.models
    # only get optimizer, not model again
    _, optimizer, _, _ = DeepHyperX.models.get_model(args.model, **kwargs)

    # if args.restore:
    #     checkpoint = load_checkpoint(args.restore)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     epoch = checkpoint['epoch'] % args.epoch + 1
    #     best_loss = checkpoint['best_loss']
    #     try:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #     except Exception as e:
    #         raise e

    print(model.__class__)

    criterion = nn.CrossEntropyLoss()

    if args.cuda is not None:
        args.cuda = 1 # in the main program, 0 specifies the 0th device, whereas in the pruning program, cuda==0 means cuda is disabled. This is a fix for that

    if args.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    compressor = Compressor(model, cuda=args.cuda)

    model.train()
    pct_pruned = 0.0
    scores = [AverageMeter() for _ in topk]

    val_scores = [0.0 for _ in topk]

    end_next_turn_flag = False
    while True:
        try:
            if end_next_turn_flag:
                break
            if epoch == 1:

                start = time.time()

                new_pct_pruned = compressor.prune(args.alpha, model=args.model)

                time_elapse = time.time() - start

                event = 'compressor.prune'
                formatted_time = str(timedelta(seconds=time_elapse))
                df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

                print("\n"+event+" took " + formatted_time + " seconds\n")
                print_memory_metrics("after compressor.prune", df_column_entry_dict)
                print('Pruned %.3f %%' % (100 * new_pct_pruned))
                start = time.time()

                top_accs = validate(model, val_loader, topk, cuda=args.cuda, df_column_entry_dict=df_column_entry_dict)

                time_elapse = time.time() - start

                event = 'validate after compressor.prune'
                formatted_time = str(timedelta(seconds=time_elapse))
                df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

                print("\n"+event+" took " + formatted_time + " seconds\n")

                print_memory_metrics("after validation following compressor.prune", df_column_entry_dict)
                # Stopping criterion
                if stop_percent == None:
                    if new_pct_pruned - pct_pruned <= 0.001 and converged(val_scores, top_accs):
                        print("Pruning percentage increased by < 0.001 and no score has become better than 0.001, quitting pruning...\n")
                        torch.save(model.state_dict(), DEEPHYPERX_PATH_LINUX + "outputs/Experiments/" + "pruneMe/" + args.model + "_alpha" + args.alpha + "_" + (kwargs['former_technique'] if kwargs['former_technique'] is not None else "") + "_" + (kwargs['former_components'] if kwargs['former_components'] is not None else ""))
                        break
                    elif new_pct_pruned - pct_pruned <= 0.001:
                        print("Pruning percentage increased by < 0.001, quitting pruning...\n")
                        torch.save(model.state_dict(), DEEPHYPERX_PATH_LINUX + "outputs/Experiments/" + "pruneMe/" + args.model + "_alpha" + args.alpha + "_" + (kwargs['former_technique'] if kwargs['former_technique'] is not None else "") + "_" + (kwargs['former_components'] if kwargs['former_components'] is not None else ""))
                        break
                else:
                    if new_pct_pruned * 100 > stop_percent:
                        print("Desired pruning percentage of "+str(stop_percent)+" reached, the new percentage would have been "+str(new_pct_pruned*100)+", quitting pruning...\n")
                        end_next_turn_flag = True

                pct_pruned = new_pct_pruned
                val_scores = top_accs

            for e in range(epoch, int(args.prune_epochs) + 1):
                for i, (input, label) in enumerate(train_loader, 0):
                    input, label = Variable(input), Variable(label)

                    if args.cuda:
                        input, label = input.cuda(), label.cuda()

                    optimizer.zero_grad()

                    output = model(input)

                    precisions = accuracy(output, label, topk)

                    for i, s in enumerate(scores):
                        s.update(precisions[i][0], input.size(0))

                    loss = criterion(output, label)
                    loss.backward()

                    compressor.set_grad()

                    optimizer.step()

                if e % kwargs['interval'] == 0:
                    checkpoint = {
                        'state_dict': model.module.state_dict()
                        if args.cuda else model.state_dict(),
                        'epoch': e,
                        'best_loss': max(best_loss, loss.item()),
                        'optimizer': optimizer.state_dict()
                    }

                    from DeepHyperX.batch_prune import MODEL_PRUNE_RESTORE_PATH
                    save_checkpoint(state=checkpoint, filename=args.model+"_alpha"+str(args.alpha)+"_pruned.pth", dir=MODEL_PRUNE_RESTORE_PATH, is_best=(loss.item() < best_loss))

                if e % 30 == 0:
                    lr = optimizer.lr * 0.1
                    adjust_learning_rate(optimizer, lr, verbose=True)

            epoch = 1

        except KeyboardInterrupt:
            """important for a number of use cases. E.g., desired pruning percentage is never reached or the steps are too slow.
            Excel file will only be generated if program runs through successfully"""
            print("execution interrupted by user's ctrl+c\n")
            end_next_turn_flag = True

    validate(model, val_loader, topk, cuda=args.cuda, df_column_entry_dict=df_column_entry_dict)