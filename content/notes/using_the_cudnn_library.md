+++
title = "Using the cuDNN Library"
authors = ["Alex Dillhoff"]
date = 2024-04-15T20:14:00-05:00
tags = ["gpgpu", "deep learning"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [What is cuDNN?](#what-is-cudnn)
- [Setting up cuDNN](#setting-up-cudnn)
- [Handling Errors](#handling-errors)
- [Representing Data](#representing-data)
- [Dense Layers](#dense-layers)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Convolutions](#convolutions)
- [Pooling](#pooling)

</div>
<!--endtoc-->



## What is cuDNN? {#what-is-cudnn}

NVIDIA cuDNN provides optimized implementations of core operations used in deep learning. It is designed to be integrated into higher-level machine learning frameworks, such as TensorFlow, PyTorch, and Caffe.


## Setting up cuDNN {#setting-up-cudnn}

To use cuDNN in your applications, each program needs to establish a handle to the cuDNN library. This is done by creating a `cudnnHandle_t` object and initializing it with `cudnnCreate`.

```C
#include <cudnn.h>

int main() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    // Use the handle
    cudnnDestroy(handle);
    return 0;
}
```


## Handling Errors {#handling-errors}

cuDNN functions return a `cudnnStatus_t` value, which indicates whether the function call was successful. As with previous CUDA code that we have reviewed, it is best to check the return value of each call. Not only does this help with debugging, but it also allows you to handle errors gracefully.


## Representing Data {#representing-data}

All data in cuDNN is represented as a **tensor**. A tensor is a multi-dimensional array of data. In cuDNN, tensors have the following parameters:

-   \# of dimensions
-   data type
-   an array of integers indicating the size of each dimension
-   an array of integers indicating the stride of each dimension

There are a few tensor descriptors for commonly used tensor types:

-   3D Tensors (BMN): Batch, Height, Width
-   4D Tensors (NCHW): Batch, Channel, Height, Width
-   5D Tensors (NCDHW): Batch, Channel, Depth, Height, Width

When creating a tensor to use with cuDNN operations, we need to create a **tensor descriptor** as well as the data itself. The tensor descriptor is a struct that contains the parameters of the tensor.

```C
// Create descriptor
cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
int n = 1, c = 3, h = 224, w = 224;
cudnnTensorDescriptor_t tensor_desc;
cudnnCreateTensorDescriptor(&tensor_desc);
cudnnSetTensor4dDescriptor(tensor_desc, format, data_type, n, c, h, w);
```

The descriptor is then used to allocate memory for the tensor data.

```C
float *data;
cudaMalloc(&data, n * c * h * w * sizeof(float));
```


### Retrieving Tensor Information {#retrieving-tensor-information}

To retrieve the properties of a tensor that already exists, we can use the `cudnnGetTensor4dDescriptor` function.

```C
cudnnStatus_t cudnnGetTensor4dDescriptor(
    const cudnnTensorDescriptor_t  tensorDesc,
    cudnnDataType_t         *dataType,
    int                     *n,
    int                     *c,
    int                     *h,
    int                     *w,
    int                     *nStride,
    int                     *cStride,
    int                     *hStride,
    int                     *wStride)
```

The parameters are as follows:

-   `tensorDesc`: the tensor descriptor
-   `dataType`: the data type of the tensor
-   `n`: the number of batches
-   `c`: the number of channels
-   `h`: the height of the tensor
-   `w`: the width of the tensor
-   `nStride`: the stride of the batch dimension
-   `cStride`: the stride of the channel dimension
-   `hStride`: the stride of the height dimension
-   `wStride`: the stride of the width dimension


## Dense Layers {#dense-layers}

A **dense layer** refers to a fully connected layer in a neural network. Each neuron in the layer is connected to every neuron in the previous layer. The weights of the connections are stored in a matrix, and the biases are stored in a vector. Implementing the forward and backward pass of a dense layer involves matrix multiplication and addition for which cuBLAS has optimized routines.


### Forward Pass {#forward-pass}

The forward pass of a dense layer is computed as follows:

\\[
\mathbf{a} = W \mathbf{x} + \mathbf{b}
\\]

where \\(\mathbf{W}\\) is the weight matrix, \\(\mathbf{x}\\) is the input tensor, \\(\mathbf{b}\\) is the bias vector, and \\(\mathbf{a}\\) is the output tensor.

This can be implemented in CUDA with a matrix multiply followed by a vector addition. The first operation we will use is [`cublasSgemm`](https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasSgemm#cublas-t-gemm). The function declaration is

```C
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc);
```

This computes

\\[
C = \alpha \text{op}(A) \text{op}(B) + \beta C.
\\]

The parameters are as follows:

-   `handle`: the cuBLAS handle
-   `transa`: the operation to perform on matrix A (transpose or not)
-   `transb`: the operation to perform on matrix B (transpose or not)
-   `m`: the number of rows in matrix A and C
-   `n`: the number of columns in matrix B and C
-   `k`: the number of columns in matrix A and rows in matrix B
-   `alpha`: scalar multiplier for the product of A and B
-   `A`: matrix A
-   `lda`: leading dimension of matrix A
-   `B`: matrix B
-   `ldb`: leading dimension of matrix B
-   `beta`: scalar multiplier for matrix C
-   `C`: matrix C
-   `ldc`: leading dimension of matrix C

This function is called twice in the forward pass of a dense layer: once for the matrix multiplication and once for the vector addition.


### Backward Pass {#backward-pass}

The backward pass of a dense layer computes the gradients of the weights and biases with respect to the loss.

\\[
\frac{\partial \mathbf{a}}{\partial W} = \frac{\partial}{\partial W} (W \mathbf{x} + \mathbf{b}) = \mathbf{x}
\\]

\\[
\frac{\partial \mathbf{a}}{\partial \mathbf{b}} = \frac{\partial}{\partial \mathbf{b}} (W \mathbf{x} + \mathbf{b}) = 1
\\]

Additionally, the layer should propagate the gradients of the loss with respect to the input tensor.

\\[
\frac{\partial \mathbf{a}}{\partial \mathbf{x}} = \frac{\partial}{\partial \mathbf{x}} (W \mathbf{x} + \mathbf{b}) = W
\\]

These gradients are only the local gradients of the layer. During backpropagation, the gradients are multiplied by the gradients propagated from the subsequent layer, as shown below:

\\[
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{a}} \frac{\partial \mathbf{a}}{\partial W}
\\]

\\[
\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{a}} \frac{\partial \mathbf{a}}{\partial \mathbf{b}}
\\]

\\[
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{a}} \frac{\partial \mathbf{a}}{\partial \mathbf{x}}
\\]

The first two gradients are used to update the weights and biases of the current layer. The last gradient is propagated to the previous layer.

These can be implemented in CUDA with matrix multiplication and vector addition, similar to the forward pass.


## Activation Functions {#activation-functions}

cuDNN supports a variety of activation functions, including:

-   Sigmoid
-   Hyperbolic Tangent
-   Rectified Linear Unit (ReLU)
-   Clipped Rectified Linear Unit (CLReLU)
-   Exponential Linear Unit (ELU)

To use an activation function, we need to create an activation descriptor and set the activation function type.

```C
cudnnActivationDescriptor_t activation_desc;
cudnnCreateActivationDescriptor(&activation_desc);
cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
```

The third enum `CUDNN_NOT_PROPAGATE_NAN` indicates that NaN values should not be propagated through the activation function. The last parameter is a coefficient value, which is used by clipped ReLU and ELU.

We can also query the activation descriptor to extract the properties.

```C
cudnnGetActivationDescriptor(activation_desc, &mode, &reluNanOpt, &coef);
```


### Forward Pass {#forward-pass}

To process the forward pass of an activation function, we use the `cudnnActivationForward` function.

```C
cudnnActivationForward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    void *alpha, // scalar multiplier
    cudnnTensorDescriptor_t xDesc,
    void *x,
    void *beta, // scalar modifier
    cudnnTensorDescriptor_t zDesc,
    void *z);
```

This computes the following operation:

\\[
\mathbf{z} = \alpha \cdot g(\mathbf{x}) + \beta \cdot \mathbf{z}
\\]

where \\(\mathbf{x}\\) is the input tensor, \\(\mathbf{z}\\) is the output tensor, \\(\alpha\\) is a scalar multiplier, and \\(\beta\\) is a scalar modifier.


### Backward Pass {#backward-pass}

Likewise, the backward pass is done with `cudnnActivationBackward`.

```C
cudnnActivationBackward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    void *alpha, // gradient modifier
    cudnnTensorDescriptor_t zDesc,
    void *z,
    cudnnTensorDescriptor_t dzDesc,
    void *dz,
    void *beta,
    cudnnTensorDescriptor_t xDesc,
    void *x,
    cudnnTensorDescriptor_t dxDesc,
    void *dx
);
```

This computes the following operation:

\\[
d\mathbf{x} = \alpha \cdot \nabla\_{\mathbf{x}} g(\mathbf{x}) d\mathbf{z} + \beta \cdot d\mathbf{x}
\\]

where \\(d\mathbf{z}\\) is the input tensor to the backward function. Under this same notation, \\(\mathbf{z}\\) was the output of the activation function. The input to the activation function \\(d\mathbf{x}\\) is the output tensor of the backward pass, since it is being propagated in the backwards direction.


## Loss Functions {#loss-functions}

cuDNN also provides optimized implementations of loss functions such as cross-entropy. Since the related lab focuses on classification, we will limit our discussion to the cross-entropy loss combined with the softmax function.


### Softmax {#softmax}

The softmax function is used to convert the output of a neural network into a probability distribution. It is defined as

\\[
\text{softmax}(\mathbf{x})\_i = \frac{e^{x\_i}}{\sum\_j e^{x\_j}}
\\]

where \\(\mathbf{x}\\) is the input tensor and \\(i\\) is the index of the output tensor.


#### Forward Pass {#forward-pass}

Implementing the forward pass of the softmax function is straightforward. We use the `cudnnSoftmaxForward` function.

```C
cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t                    handle,
    cudnnSoftmaxAlgorithm_t          algorithm,
    cudnnSoftmaxMode_t               mode,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *beta,
    const cudnnTensorDescriptor_t    yDesc,
    void                            *y)
```

Most of the parameters are similar to other cuDNN functions. The `algorithm` parameter specifies the algorithm to use for the softmax function, and the `mode` parameter specifies the mode of the softmax function.

-   `algorithm`: `CUDNN_SOFTMAX_FAST`, `CUDNN_SOFTMAX_ACCURATE`, or `CUDNN_SOFTMAX_LOG`. The most numerically stable is `CUDNN_SOFTMAX_ACCURATE`.
-   `mode`: `CUDNN_SOFTMAX_MODE_INSTANCE` or `CUDNN_SOFTMAX_MODE_CHANNEL`. The former computes the softmax function for each instance in the batch, while the latter computes the softmax function for each channel in the tensor.


#### Backward Pass {#backward-pass}

The backward pass of the softmax function is implemented with `cudnnSoftmaxBackward`.

```C
cudnnStatus_t cudnnSoftmaxBackward(
    cudnnHandle_t                    handle,
    cudnnSoftmaxAlgorithm_t          algorithm,
    cudnnSoftmaxMode_t               mode,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    yDesc,
    const void                      *y,
    const cudnnTensorDescriptor_t    dyDesc,
    const void                      *dy,
    const void                      *beta,
    const cudnnTensorDescriptor_t    dxDesc,
    void                            *dx)
```

Note that the softmax function is used in the forward pass of the loss function, so the gradients are propagated from the loss function to the softmax function. In practice, the two are combined into a much simpler gradient calculation. If the softmax function is followed by the cross-entropy loss, the gradients are computed as

\\[
\frac{\partial L}{\partial \mathbf{x}} = \mathbf{x} - \mathbf{y}
\\]

where \\(\mathbf{y}\\) is the target tensor.


## Convolutions {#convolutions}

For a background on convolutions, see [these notes]({{< relref "convolutional_neural_networks.md" >}}). The notes in this article refer to the cuDNN implementation of convolutions.

When using convolution operations in cuDNN, we need to create a convolution descriptor `cudnnConvolutionDescriptor_t` as well as a filter descriptor `cudnnFilterDescriptor_t`.


### Creating a filter {#creating-a-filter}

To create a filter descriptor, we use the `cudnnCreateFilterDescriptor` function.

```C
cudnnFilterDescriptor_t filter_desc;
cudnnCreateFilterDescriptor(&filter_desc);
```

We then set the filter descriptor with the `cudnnSetFilter4dDescriptor` function.

```C
cudnnStatus_t cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t    filterDesc,
    cudnnDataType_t            dataType,
    cudnnTensorFormat_t        format,
    int                        k,
    int                        c,
    int                        h,
    int                        w)
```

The parameters are as follows:

-   `filterDesc`: the filter descriptor
-   `dataType`: the data type of the filter
-   `format`: the format of the filter (NCHW or NHWC). Use `CUDNN_TENSOR_NCHW` for most cases.
-   `k`: the number of output feature maps
-   `c`: the number of input feature maps
-   `h`: the height of the filter
-   `w`: the width of the filter

We can also query the filter descriptor to extract the properties.

```C
cudnnDataType_t data_type;
cudnnTensorFormat_t format;
int k, c, h, w;
cudnnGetFilter4dDescriptor(filter_desc, &data_type, &format, &k, &c, &h, &w);
```


### Creating a convolution {#creating-a-convolution}

To create a convolution descriptor, we use the `cudnnCreateConvolutionDescriptor` function. Once we are done with it, we should destroy it with `cudnnDestroyConvolutionDescriptor`. Since our convolution is 2D, we use the `cudnnSetConvolution2dDescriptor` function.

```C
cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t    convDesc,
    int                             pad_h,
    int                             pad_w,
    int                             u,
    int                             v,
    int                             dilation_h,
    int                             dilation_w,
    cudnnConvolutionMode_t          mode,
    cudnnDataType_t                 computeType)
```

The parameters are as follows:

-   `convDesc`: the convolution descriptor
-   `pad_h`: the height padding
-   `pad_w`: the width padding
-   `u`: the vertical stride
-   `v`: the horizontal stride
-   `dilation_h`: the height dilation
-   `dilation_w`: the width dilation
-   `mode`: the convolution mode (`CUDNN_CONVOLUTION` or `CUDNN_CROSS_CORRELATION`)
-   `computeType`: the data type used for the convolution

Although the library supports both convolution and cross-correlation, the difference is only in the order of the operands. In practice, the two are equivalent. Most deep learning frameworks use cross-correlation.

To query the convolution descriptor, we can use the `cudnnGetConvolution2dDescriptor` function.

```C
cudnnStatus_t cudnnGetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t    convDesc,
    int                            *pad_h,
    int                            *pad_w,
    int                            *u,
    int                            *v,
    int                            *dilation_h,
    int                            *dilation_w,
    cudnnConvolutionMode_t         *mode,
    cudnnDataType_t                *computeType)
```

If we know all the parameters of the convolution, we can use the `cudnnGetConvolution2dForwardOutputDim` function to calculate the output dimensions.

```C
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t    convDesc,
    const cudnnTensorDescriptor_t         inputTensorDesc,
    const cudnnFilterDescriptor_t         filterDesc,
    int                                 *n,
    int                                 *c,
    int                                 *h,
    int                                 *w)
```


### Forward Pass {#forward-pass}

`cuDNN` supports several methods for performing a convolution operation. An evaluation of the available algorithms can be found [here.](https://core.ac.uk/download/pdf/224976536.pdf) The algorithms provide tradeoffs in terms of speed and memory usage. Diving into these details is beyond the scope of this article, but it is important to be aware of the options.

The forward pass of a convolution is implemented with the `cudnnConvolutionForward` function.

```C
cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionFwdAlgo_t           algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       yDesc,
    void                               *y)
```

The parameters are as follows:

-   `handle`: the cuDNN handle
-   `alpha`: scalar multiplier for the input tensor
-   `xDesc`: the input tensor descriptor
-   `x`: the input tensor
-   `wDesc`: the filter descriptor
-   `w`: the filter tensor
-   `convDesc`: the convolution descriptor
-   `algo`: the algorithm to use for the convolution
-   `workSpace`: the workspace for the convolution
-   `workSpaceSizeInBytes`: the size of the workspace
-   `beta`: scalar modifier for the output tensor
-   `yDesc`: the output tensor descriptor
-   `y`: the output tensor


### Backward Pass {#backward-pass}

There are three different backward passes for a convolutional layer: one for the weights, one for the input tensor, and one for the bias.


#### Weights {#weights}

The backward pass for the weights is implemented with the `cudnnConvolutionBackwardFilter` function.

```C
cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnTensorDescriptor_t       dyDesc,
    const void                         *dy,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionBwdFilterAlgo_t     algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnFilterDescriptor_t       dwDesc,
    void                               *dw)
```

A detailed description of the parameters can be found [here.](https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionbackwardfilter)


#### Bias {#bias}

The backward pass for the bias is implemented with the `cudnnConvolutionBackwardBias` function.

```C
cudnnStatus_t cudnnConvolutionBackwardBias(
    cudnnHandle_t                    handle,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    dyDesc,
    const void                      *dy,
    const void                      *beta,
    const cudnnTensorDescriptor_t    dbDesc,
    void                            *db)
```

A detailed description of the parameters can be found [here.](https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionbackwardbias)


#### Input {#input}

The backward pass for the input tensor is implemented with the `cudnnConvolutionBackwardData` function.

```C
cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnTensorDescriptor_t       dyDesc,
    const void                         *dy,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionBwdDataAlgo_t       algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       dxDesc,
    void                               *dx)
```

A detailed description of the parameters can be found [here](https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionbackwarddata).


## Pooling {#pooling}

Pooling is a common operation in convolutional neural networks. It reduces the spatial dimensions of the input tensor, which helps to reduce the number of parameters and computation in the network. Using pooling in cuDNN requires creating a descriptor. Make sure to destroy it when you're done.


### Creating a pooling descriptor {#creating-a-pooling-descriptor}

To create a pooling descriptor, we use the `cudnnCreatePoolingDescriptor` function.

```C
cudnnPoolingDescriptor_t pooling_desc;
cudnnCreatePoolingDescriptor(&pooling_desc);
```

We then set the pooling descriptor with the `cudnnSetPooling2dDescriptor` function.

```C
cudnnStatus_t cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t    poolingDesc,
    cudnnPoolingMode_t          mode,
    cudnnNanPropagation_t       maxpoolingNanOpt,
    int                         windowHeight,
    int                         windowWidth,
    int                         verticalPadding,
    int                         horizontalPadding,
    int                         verticalStride,
    int                         horizontalStride)
```

The parameters are as follows:

-   `poolingDesc`: the pooling descriptor
-   `mode`: the pooling mode (`CUDNN_POOLING_MAX` or `CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING`)
-   `maxpoolingNanOpt`: the NaN propagation option for max pooling
-   `windowHeight`: the height of the pooling window
-   `windowWidth`: the width of the pooling window
-   `verticalPadding`: the vertical padding
-   `horizontalPadding`: the horizontal padding
-   `verticalStride`: the vertical stride
-   `horizontalStride`: the horizontal stride

We can also query the pooling descriptor to extract the properties.

```C
cudnnPoolingMode_t mode;
cudnnNanPropagation_t nan_opt;
int window_h, window_w, pad_h, pad_w, stride_h, stride_w;
cudnnGetPooling2dDescriptor(pooling_desc, &mode, &nan_opt, &window_h, &window_w, &pad_h, &pad_w, &stride_h, &stride_w);
```


### Forward Pass {#forward-pass}

The forward pass of a pooling operation is implemented with the `cudnnPoolingForward` function.

```C
cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t                    handle,
    const cudnnPoolingDescriptor_t   poolingDesc,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *beta,
    const cudnnTensorDescriptor_t    yDesc,
    void                            *y)
```

The parameters are as follows:

-   `handle`: the cuDNN handle
-   `poolingDesc`: the pooling descriptor
-   `alpha`: scalar multiplier for the input tensor
-   `xDesc`: the input tensor descriptor
-   `x`: the input tensor
-   `beta`: scalar modifier for the output tensor
-   `yDesc`: the output tensor descriptor
-   `y`: the output tensor


### Backward Pass {#backward-pass}

The backward pass of a pooling operation is implemented with the `cudnnPoolingBackward` function.

```C
cudnnStatus_t cudnnPoolingBackward(
    cudnnHandle_t                    handle,
    const cudnnPoolingDescriptor_t   poolingDesc,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    yDesc,
    const void                      *y,
    const cudnnTensorDescriptor_t    dyDesc,
    const void                      *dy,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *beta,
    const cudnnTensorDescriptor_t    dxDesc,
    void                            *dx)
```

The parameters are as follows:

-   `handle`: the cuDNN handle
-   `poolingDesc`: the pooling descriptor
-   `alpha`: scalar multiplier for the input tensor
-   `yDesc`: the output tensor descriptor
-   `y`: the output tensor
-   `dyDesc`: the input tensor descriptor
-   `dy`: the input tensor
-   `xDesc`: the input tensor descriptor
-   `x`: the input tensor
-   `beta`: scalar modifier for the output tensor
-   `dxDesc`: the output tensor descriptor
-   `dx`: the output tensor
