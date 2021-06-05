### Knowledge Distillation

Implement a basic knowledge distillation model with tinynn.

- teacher network

    ```
    Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1])
    MaxPool2D(pool_size=[2, 2], stride=[2, 2])
    Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1])
    MaxPool2D(pool_size=[2, 2], stride=[2, 2])
    Flatten()
    Dense(120)
    Dense(84)
    Dense(10)
    ```

- student network

    ```
    Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1])
    Flatten()
    Dense(120)
    Dense(10)
    ```

- dataset: Fashion-MNIST


#### Results

![result](https://tva1.sinaimg.cn/large/006y8mN6ly1g8t8pu69zfj30d8084aam.jpg)

