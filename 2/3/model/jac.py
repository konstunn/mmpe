
import tensorflow as tf

x = [2., 3.]
s = len(x)
x = tf.constant(x)

a = [[2*x[0]*x[1], 3*x[0]*x[1]],
     [4*x[0]*x[1], 5*x[0]*x[1]]]

a = tf.convert_to_tensor(a)

n = tf.size(a)

a = tf.reshape(a, [n])


def cond(i, N, rez):
    return tf.less(i, N)


def body(i, N, rez):
    elem = tf.slice(a, [i], [1])
    elem = tf.gradients(elem, x)
    elem = tf.reshape(elem, [s, 1])
    rez = tf.concat([rez, elem], 1)
    i = i + 1
    return i, N, rez


shape_invariants = [
                    n.get_shape(),
                    n.get_shape(),
                    tf.TensorShape([s, None])
]

elem = tf.slice(a, [0], [1])
rez = tf.gradients(elem, x)[0]
rez = tf.reshape(rez, [s, 1])

loop = tf.while_loop(cond, body, [1, n, rez], shape_invariants)
