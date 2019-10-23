"""Network layers and activation layers."""

import numpy as np

from core.initializer import XavierUniform
from core.initializer import Zeros
from core.initializer import Ones


class Layer(object):

    def __init__(self):
        self.params, self.grads = {}, {}
        self.is_training = True
        self.shapes = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def set_phase(self, phase):
        self.is_training = True if phase == "TRAIN" else False

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return "layer: %s \t shape: %s" % (self.name, self.shapes)


class Dense(Layer):

    def __init__(self,
                 num_out,
                 num_in=None,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": [num_in, num_out], "b": [num_out]}
        self.params = {"w": None, "b": None}

        self.is_init = False
        if num_in is not None:
            self._init_params(num_in)

        self.inputs = None

    def forward(self, inputs):
        # lazy initialize
        if not self.is_init:
            self._init_params(inputs.shape[1])

        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T

    def _init_params(self, input_size):
        self.shapes["w"][0] = input_size
        self.params["w"] = self.initializers["w"](shape=self.shapes["w"])
        self.params["b"] = self.initializers["b"](shape=self.shapes["b"])
        self.is_init = True


class Conv2D(Layer):
    """
    Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (height, width,
        in_channels, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: weight initializer
    :param b_init: bias initializer
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": self.kernel_shape, "b": self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None

        self.is_init = False

    def forward(self, inputs):
        """
        Accelerate convolution via im2col trick.
        An example (assuming only one channel and one filter):
         input = | 43  16  78 |         kernel = | 4  6 |
          (X)    | 34  76  95 |                  | 7  9 |
                 | 35   8  46 |
        
        After im2col and kernel flattening:
         col  = | 43  16  34  76 |     kernel = | 4 |
                | 16  78  76  95 |      (W)     | 6 |
                | 34  76  35   8 |              | 7 |
                | 76  95   8  46 |              | 9 |
        """
        # lazy initialization
        if not self.is_init:
            self._init_params()

        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)

        # step2: padded inputs to column matrix
        self.col = im2col(X, k_h, k_w, s_h, s_w)

        # step3: perform convolution by matrix product.
        self.W = self.params["w"].reshape(-1, out_c)  # (k_h * k_w * in_c, out_c)
        Z = self.col @ self.W

        # step4: reshape output 
        batch_sz, in_h, in_w, _ = X.shape
        # separate the batch size and feature map dimensions
        Z = Z.reshape(batch_sz, Z.shape[0] // batch_sz, out_c)
        # further divide the feature map in to (h, w) dimension
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        Z = Z.reshape(batch_sz, out_h, out_w, out_c)

        # plus the bias for every filter
        Z += self.params["b"]
        # save results for backward function
        self.X_shape = X.shape
        return Z

    def backward(self, grad):
        """
        Compute gradients w.r.t layer parameters and backward gradients.
        :param grad: gradients from previous layer 
            with shape (batch_sz, out_h, out_w, out_c)
        :return d_in: gradients to next layers 
            with shape (batch_sz, in_h, in_w, in_c)
        """
        # read size parameters
        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.X_shape
        pad_h, pad_w = self.padding[1:3]

        # calculate gradients of parameters
        flat_grad = grad.reshape((-1, out_c))
        d_W = self.col.T @ flat_grad
        self.grads["w"] = d_W.reshape(self.kernel_shape)
        self.grads["b"] = np.sum(flat_grad, axis=0)

        # calculate backward gradients
        d_X = grad @ self.W.T
        # cast gradients back to original shape as d_in
        d_in = np.zeros(shape=self.X_shape)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = d_X[:, i, j, :]
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r:r+k_h, c:c+k_w, :] += patch

        # cut off gradients of padding
        d_in = d_in[:, pad_h[0]:in_h-pad_h[1], pad_w[0]:in_w-pad_w[1], :]
        return self._grads_postprocess(d_in)

    def _inputs_preprocess(self, inputs):
        _, in_h, in_w, _ = inputs.shape
        k_h, k_w, _, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def _grads_postprocess(self, grads):
        return grads

    def _init_params(self):
        self.params["w"] = self.initializers["w"](self.kernel_shape)
        self.params["b"] = self.initializers["b"](self.kernel_shape[-1])
        self.is_init = True


class MaxPool2D(Layer):

    def __init__(self,
                 pool_size,
                 stride,
                 padding="VALID"):
        """
        Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__()
        self.shape = None

        self.kernel_shape = pool_size
        self.stride = stride

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_h, in_w, in_c = inputs.shape

        # zero-padding
        if self.padding is None:
            self.padding = get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")
        padded_h, padded_w = X.shape[1:3]
    
        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # construct output matrix and argmax matrix
        max_pool = np.empty(shape=(batch_sz, out_h, out_w, in_c))
        argmax = np.empty(shape=(batch_sz, out_h, out_w, in_c), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, r_start: r_start+k_h, c_start: c_start+k_w, :]
                pool = pool.reshape((batch_sz, -1, in_c))

                _argmax = np.argmax(pool, axis=1)[:, np.newaxis, :]
                argmax[:, r, c, :] = _argmax.squeeze()

                # get max elements
                _max_pool = np.take_along_axis(pool, _argmax, axis=1).squeeze()
                max_pool[:, r, c, :] = _max_pool

        self.X_shape = X.shape
        self.out_shape = (out_h, out_w)
        self.argmax = argmax
        return max_pool

    def backward(self, grad):
        batch_sz, in_h, in_w, in_c = self.X_shape
        out_h, out_w = self.out_shape
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        k_sz = k_h * k_w
        pad_h, pad_w = self.padding[1:3]

        d_in = np.zeros(shape=(batch_sz, in_h, in_w, in_c))
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                _argmax = self.argmax[:, r, c, :]
                mask = np.eye(k_sz)[_argmax].transpose((0, 2, 1))
                _grad = grad[:, r, c, :][:, np.newaxis, :]
                patch = np.repeat(_grad, k_sz, axis=1) * mask
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r_start:r_start+k_h, c_start:c_start+k_w, :] += patch

        # cut off gradients of padding
        d_in = d_in[:, pad_h[0]:in_h-pad_h[1], pad_w[0]:in_w-pad_w[1], :]
        return d_in


class ConvTranspose2D(Conv2D):

    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__(kernel, stride, padding, w_init, b_init)
        self.origin_stride = stride
        self.stride = (1, 1)

    def _inputs_preprocess(self, inputs):
        k_h, k_w = self.kernel_shape[:2]

        # insert zeros to inputs
        inputs = self._insert_zeros(inputs,
                                    *self.origin_stride,
                                    self.padding_mode)
        batch_sz, in_h, in_w, in_c = inputs.shape

        # padding calculation
        if self.padding is None:
            if self.padding_mode == "SAME":
                self.padding = get_padding_2d(
                    (in_h, in_w), (k_h, k_w), self.padding_mode)
            else:
                self.padding = ((0, 0), (k_h - 1, k_h - 1),
                                (k_w - 1, k_w - 1), (0, 0))
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def _grads_postprocess(self, grads):
        return grads[:, ::self.origin_stride[0], ::self.origin_stride[1], :]

    @staticmethod
    def _insert_zeros(inputs, s_h, s_w, mode):
        batch_sz, in_h, in_w, in_c = inputs.shape

        if mode == "SAME":
            out_h = in_h * s_h
            out_w = in_w * s_w
        else:
            out_h = (in_h - 1) * s_h + 1
            out_w = (in_w - 1) * s_h + 1
        expand = np.zeros((batch_sz, out_h, out_w, in_c))
        expand[:, ::s_h, ::s_w, :] = inputs
        return expand


class BatchNormalization(Layer):

    def __init__(self,
                 momentum=0.99,
                 gamma_init=Ones(),
                 beta_init=Zeros(),
                 running_mean_init=Zeros(),
                 running_var_init=Ones(),
                 epsilon=1e-3):
        super().__init__()
        self.m = momentum
        self.epsilon = epsilon

        self.running_mean = None
        self.running_var = None

        self.initializer = {"gamma": gamma_init, "beta": beta_init,
                            "running_mean": running_mean_init,
                            "running_var": running_var_init}
        self.params = {"gamma": None, "beta": None}
        self.shapes = {"gamma": None, "beta": None}

        self.is_init = False

    def forward(self, inputs):
        # lazy initialize
        if not self.is_init:
            self._init_params(inputs.shape[1:])

        if self.is_training:
            mean = inputs.mean(axis=0)
            var = inputs.var(axis=0)
            self.running_mean = self.m * self.running_mean + (1 - self.m) * mean
            self.running_var = self.m * self.running_var + (1 - self.m) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # normalize
        self.X_center = inputs - mean
        self.std = (var + self.epsilon) ** 0.5
        X = self.X_center / self.std
        return self.params["gamma"] * X + self.params["beta"]

    def backward(self, grad):
        pass

    def _init_params(self, input_size):
        for param in ["gamma", "beta"]:
            self.params[param] = self.initializer[param](input_size)
            self.shapes[param] = input_size

        self.running_mean = self.initializer["running_mean"](input_size)
        self.running_var = self.initializer["running_var"](input_size)
        self.is_init = True


class Reshape(Layer):

    def __init__(self, *output_shape):
        super().__init__()
        self.input_shape = None
        self.output_shape = output_shape

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.output_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class Flatten(Reshape):

    def __init__(self):
        super().__init__(-1)


class Dropout(Layer):

    def __init__(self, keep_prob=0.5):
        super().__init__()
        self._keep_prob = keep_prob
        self._multiplier = None

    def forward(self, inputs):
        if self.is_training:
            multiplier = np.random.binomial(
                1, self._keep_prob, size=inputs.shape)
            self._multiplier = multiplier / self._keep_prob
            outputs = inputs * self._multiplier
        else:
            outputs = inputs
        return outputs

    def backward(self, grad):
        assert self.is_training is True
        return grad * self._multiplier


class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_func(self, x):
        return self.func(x) * (1.0 - self.func(x))


class Softplus(Activation):

    def func(self, x):
        return np.log(1.0 + np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def derivative_func(self, x):
        return 1.0 / (1.0 + np.exp(-x))


class Tanh(Activation):

    def func(self, x):
        return np.tanh(x)

    def derivative_func(self, x):
        return 1.0 - self.func(x) ** 2


class ReLU(Activation):

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative_func(self, x):
        return x > 0.0


class LeakyReLU(Activation):

    def __init__(self, slope=0.2):
        super().__init__()
        self._slope = slope

    def func(self, x):
        x = x.copy()
        x[x < 0.0] *= self._slope
        return x

    def derivative_func(self, x):
        dx = np.ones_like(x)
        dx[x < 0.0] = self._slope
        return dx


def im2col(img, k_h, k_w, s_h, s_w):
    """
    Transform padded image into column matrix.
    :param img: padded inputs of shape (B, in_h, in_w, in_c)
    :param k_h: kernel height
    :param k_w: kernel width
    :param s_h: stride height
    :param s_w: stride width
    :return col: column matrix of shape (B*out_h*out_w, k_h*k_h*inc)
    """
    batch_sz, h, w, in_c = img.shape
    # calculate result feature map size
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    # allocate space for column matrix
    col = np.empty((batch_sz * out_h * out_w, k_h * k_w * in_c))
    # fill in the column matrix
    batch_span = out_w * out_h
    for r in range(out_h):
        r_start = r * s_h
        matrix_r = r * out_w
        for c in range(out_w):
            c_start = c * s_w
            patch = img[:, r_start: r_start+k_h, c_start: c_start+k_w, :]
            patch = patch.reshape(batch_sz, -1)
            col[matrix_r+c::batch_span, :] = patch
    return col


def get_padding_2d(in_shape, k_shape, mode):
    def get_padding_1d(w, k):
        if mode == "SAME":
            pads = (w - 1) + k - w
            half = pads // 2
            padding = (half, half) if pads % 2 == 0 else (half, half + 1)
        else:
            padding = (0, 0)
        return padding

    h_pad = get_padding_1d(in_shape[0], k_shape[0])
    w_pad = get_padding_1d(in_shape[1], k_shape[1])
    return (0, 0), h_pad, w_pad, (0, 0)
