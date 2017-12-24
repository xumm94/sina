import tensorflow as tf
import numpy as np
from fastText_model import fastText
from data_util import load_data
from tflearn.data_utils import  pad_sequences
import pickle
import os

'''
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size",10,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"num_epochs")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
'''

def do_eval(sess,fast_text,evalX,evalY,batch_size):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([fast_text.loss_val, fast_text.accuracy],
                                          feed_dict={fast_text.sentence: evalX[start:end],fast_text.labels: evalY[start:end]})
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)


def main():
    train_x, train_y, val_x, val_y, test_x, test_y, vocab_size = load_data()

    label_size = 10
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000
    decay_rate = 0.8
    ckpt_dir = "fast_text_checkpoint/"
    sentence_len = 200
    embed_size = 100
    is_training = True
    num_epochs = 15
    validate_every = 1


    print("start padding...")

    train_x = pad_sequences(train_x, maxlen=sentence_len, value = 0)
    val_x = pad_sequences(val_x, maxlen=sentence_len, value = 0)
    test_x = pad_sequences(test_x, maxlen=sentence_len, value=0)
    print("end padding...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:

        fast_text = fastText(label_size = 10,
                             learning_rate = 0.01,
                             batch_size = 128,
                             decay_step = 20000,
                             decay_rate = 0.8,
                             sentence_len =  200,
                             vocab_size = vocab_size,
                             embed_size = 100,
                             is_training = True)

        saver = tf.train.Saver()
        if os.path.exists(ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())

        curr_epoch = sess.run(fast_text.epoch_step)

        number_of_training_data = len(train_x)
        batch_size = batch_size

        for epoch in range(curr_epoch, num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):

                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:",train_x[start:end].shape)
                    print("trainY[start:end]:",train_y[start:end].shape)


                curr_loss, curr_acc, _ = sess.run([fast_text.loss_val, fast_text.accuracy, fast_text.train_op],
                                                  feed_dict= \
                                                      {   fast_text.sentence : train_x[start : end],
                                                          fast_text.labels : train_y[start : end]}
                                                  )
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1

                if counter % 500 == 0:
                    print(epoch)
                    print(counter)
                    print(loss)
                    print(acc)
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (epoch, counter, loss / float(counter), acc / float(counter)))

            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            print(epoch, validate_every, (epoch % validate_every == 0))

            if epoch % validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fast_text, val_x, val_y, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))

                # save model to checkpoint
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=fast_text.epoch_step)  # fast_text.epoch_step

        test_loss, test_acc = do_eval(sess, fast_text, test_x, test_y, batch_size)
        print("test Loss:%.3f\ttest Accuracy: %.3f" % (test_loss, test_acc))
    return



if __name__ == "__main__":
    main()