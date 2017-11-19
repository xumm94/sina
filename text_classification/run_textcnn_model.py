'''
from TextCNN_model import *
from data.cnews_loader import *
from sklearn import metrics
import sys
import os

import time
from datetime import timedelta


base_dir = 'data/cnews'
train_dir = 'data/cnews/cnews.train.txt'
test_dir = 'data/cnews/cnews.test.txt'
val_dir = 'data/cnews/cnews.val.txt'
vocab_dir = 'data/cnews/cnews.vocab.txt'

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds= int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x : x_batch,
        model.input_y : y_batch,
        model.keep_prob : keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, keep_prob= 1.0)
        batch_loss, batch_acc = sess.run([model.loss, model.acc], feed_dict = feed_dict)
        total_loss += batch_len * batch_loss
        total_acc += batch_len * batch_acc

    loss = total_loss / data_len
    acc = total_acc / data_len

    return loss, acc


def train():
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir)

    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = 'tesorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Loading training and validation data...")

    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    session = tf.Session(config = sess_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')

    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000

    flag = False

    def train():
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir)

        print("Configuring TensorBoard and Saver...")
        tensorboard_dir = 'tesorboard/textcnn'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('accuracy', model.loss)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)

        saver = tf.train.Saver()
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print("Loading training and validation data...")

        start_time = time.time()
        x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
        x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        session = tf.Session(config=sess_config)
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        print('Training and evaluating...')

        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improvement = 1000

        flag = False

        for epoch in range(config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, config.drop_keep_prob)

                if total_batch % config.save_per_batch == 0:
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % config.print_per_batch == 0:

                    feed_dict[model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([model.loss, model.acc],
                                                        feed_dict=feed_dict)
                    loss_val, acc_val = evaluate(session, x_val, y_val)

                    if (acc_val > best_acc_val):
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(model.optim, feed_dict=feed_dict)
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    session = tf.Session(config= sess_config)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pre, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TextCNNConfig()
    categories, cat_to_id = read_category()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
'''

from TextCNN_model import *
from data.cnews_loader import *
from sklearn import metrics
import sys
import os

import time
from datetime import timedelta


base_dir = 'data/cnews'
train_dir = 'data/cnews/cnews.train.txt'
test_dir = 'data/cnews/cnews.test.txt'
val_dir = 'data/cnews/cnews.val.txt'
vocab_dir = 'data/cnews/cnews.vocab.txt'

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds= int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x : x_batch,
        model.input_y : y_batch,
        model.keep_prob : keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, keep_prob= 1.0)
        batch_loss, batch_acc = sess.run([model.loss, model.acc], feed_dict = feed_dict)
        total_loss += batch_len * batch_loss
        total_acc += batch_len * batch_acc

    loss = total_loss / data_len
    acc = total_acc / data_len

    return loss, acc


def train():
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir)

    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = 'tesorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Loading training and validation data...")

    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    session = tf.Session(config = sess_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')

    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000

    flag = False

    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:

                s = session.run(merged_summary, feed_dict= feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:

                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc],
                                                    feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)

                if(acc_val > best_acc_val):
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path= save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict= feed_dict)
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    session = tf.Session(config= sess_config)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pre, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)



if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TextCNNConfig()
    categories, cat_to_id = read_category()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()



