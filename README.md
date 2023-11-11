# go-micrograd

![gopher](gopher.jpg)

This is a Go implementation of Andre Karpathy's micrograd library. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API (as close as you can get to it in Go). Both are tiny, with about 100-200 lines of code each. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the `main.go` file shows. Potentially useful for educational purposes.

### Example usage
```go
package main

import "github.com/nathan-barry/go-micrograd"

func main() {
    // inputs x1, x2
    x1 := New(2.0)
    x2 := New(0.0)

    // weights w1, w2
    w1 := New(-3.0)
    w2 := New(1.0)

    // bias
    b := New(6.8813735870195432)

    // x1*w1 + x2*w2 + b
    w1x1 := Mul(w1, x1)
    w2x2 := Mul(w2, x2)
    w1x1w2x2 := Add(w1x1, w2x2)
    n := Add(w1x1w2x2, b)

    // activation function
    o := Tanh(n)

    // backward pass
    o.Backward()

    // display computation graph
    o.DisplayGraph()
}
```

### Training a neural net

The `main.go` file has various examples training a neural network, specifically `func example3()`. Add this into `main()` to run the example.

```go
func example3()
    // Initialize the MultiLayer Perceptron
    n := NewMLP(3, []int{4, 4, 1})

    // Create the input dataset and correct outputs
    xs := [][]*Value{
        {New(2), New(3), New(-1)},
        {New(3), New(-1), New(0.5)},
        {New(0.5), New(1), New(1)},
        {New(1), New(1), New(-1)},
    }
    ys := []*Value{New(1), New(-1), New(-1), New(1)}

    // Training loop
    for k := 0; k < 50; k++ {

        // forward pass
        ypred := make([]*Value, 4)
        for i, x := range xs {
            ypred[i] = n.Forward(x)[0]
        }
        loss := MSE(ypred, ys)

        // backwards pass
        for _, p := range n.Parameters() {
            p.Grad = 0
        }
        loss.Backward()

        // update weights
        for _, p := range n.Parameters() {
            p.Data += -0.1 * p.Grad
        }

        fmt.Printf("Iter: %2v, Loss: %v\n", k, loss.Data)
    }
}
```

### License

MIT
