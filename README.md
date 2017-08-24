# caffe-with-kn2row-optimization
A temporary modified caffe framework to implement the kn2row optimization in base-conv layer

Here is the corresponding paper: (Parallel Multi Channel Convolution using General Matrix Multiplication)

# usage
to use the ker2row optimization, just specified the "optimization" and "using_approximate" field in the conv_param

one example:
```
layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    optimization: KN2ROW
    using_approximate: true
  }
}
```
to use im2col optimization, just replace `KN2ROW` with `IM2COL` or delete this field in prototxt file, the defualt value is `IM2COL`. To obtain approximate results in ker2row mode, set using_approximate to be `true` otherwise set it to be `false` to obtain precise results. The using_approximate field is useful only when optimzation is `KN2ROW`.

# note
the current optimization is only implemented in forward operations and it does not support model training.
