package main

import (
	"fmt"
	"math"
	"math/rand"
)

func main() {
	// // inputs x1, x2
	// x1 := New(2.0)
	// x2 := New(0.0)

	// // weights w1, w2
	// w1 := New(-3.0)
	// w2 := New(1.0)

	// // // bias
	// // b := New(6.8813735870195432)

	// // // x1*w1 + x2*w2 + b
	// // w1x1 := Mul(w1, x1)
	// // w2x2 := Mul(w2, x2)
	// // w1x1w2x2 := Add(w1x1, w2x2)
	// // n := Add(w1x1w2x2, b)

	// // // Tanh
	// // o := Tanh(n)
	// // // e := Exp(Mul(New(2), n))
	// // // o := Div(Sub(e, New(1)), Add(e, New(1)))

	// // o.Backward()

	// // o.DisplayGraph()

	x := []*Value{New(2), New(3), New(-1)}
	n := NewMLP(3, []int{4, 4, 1})
	out := n.Forward(x)
	out[0].Backward()

	out[0].DisplayGraph()
}

// Value struct

type Value struct {
	Data     float64
	Grad     float64
	backward func()
	prev     []*Value
	op       string
}

func New(f float64) *Value {
	return &Value{Data: f}
}

// Operators

func Add(a, b *Value) *Value {
	out := &Value{
		Data: a.Data + b.Data,
		prev: []*Value{a, b},
		op:   "+",
	}
	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}

	return out
}

func Neg(x *Value) *Value {
	return Mul(x, New(-1))
}

func Sub(a, b *Value) *Value {
	return Add(a, Neg(b))
}

func Mul(a, b *Value) *Value {
	out := &Value{
		Data: a.Data * b.Data,
		prev: []*Value{a, b},
		op:   "*",
	}
	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}

	return out
}

func Div(a, b *Value) *Value {
	return Mul(a, Pow(b, New(-1)))
}

func Pow(a, b *Value) *Value {
	out := &Value{
		Data: math.Pow(a.Data, b.Data),
		prev: []*Value{a, b},
		op:   "Pow",
	}
	out.backward = func() {
		a.Grad += (b.Data * math.Pow(a.Data, b.Data-1)) * out.Grad
		b.Grad += (a.Data * math.Pow(b.Data, a.Data-1)) * out.Grad
	}

	return out
}

func Tanh(x *Value) *Value {
	out := &Value{
		Data: (math.Exp(2*x.Data) - 1) / (math.Exp(2*x.Data) + 1),
		prev: []*Value{x},
		op:   "tanh",
	}
	out.backward = func() {
		x.Grad += (1 - math.Pow(out.Data, 2)) * out.Grad
	}

	return out
}

func Exp(x *Value) *Value {
	out := &Value{
		Data: math.Exp(x.Data),
		prev: []*Value{x},
		op:   "exp",
	}
	out.backward = func() {
		x.Grad += out.Data * out.Grad
	}

	return out
}

// Backprop

func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	topo = buildTopo(v, topo, visited)

	v.Grad = 1.0

	for i := len(topo) - 1; i >= 0; i-- {
		if len(topo[i].prev) != 0 {
			topo[i].backward()
		}
	}
}

func buildTopo(v *Value, topo []*Value, visited map[*Value]bool) []*Value {
	if !visited[v] {
		visited[v] = true
		for _, prev := range v.prev {
			topo = buildTopo(prev, topo, visited)
		}
		topo = append(topo, v)
	}
	return topo
}

// Neural Networks

type Neuron struct {
	Weight []*Value
	Bias   *Value
}

func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := 0; i < nin; i++ {
		w[i] = New(2*rand.Float64() - 1)
	}
	b := New(2*rand.Float64() - 1)

	return &Neuron{
		Weight: w,
		Bias:   b,
	}
}

func (n *Neuron) Forward(x []*Value) *Value {
	if len(n.Weight) != len(x) {
		panic("Weight and data diff dimensions")
	}

	act := n.Bias
	for i := 0; i < len(x); i++ {
		act = Add(act, Mul(n.Weight[i], x[i]))
	}

	out := Tanh(act)

	return out
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = NewNeuron(nin)
	}

	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(x)
	}

	return out
}

type MLP struct {
	Sizes  []int
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sizes := append([]int{nin}, nouts...)
	layers := make([]*Layer, len(nouts))

	for i := 0; i < len(nouts); i++ {
		layers[i] = NewLayer(sizes[i], sizes[i+1])
	}

	return &MLP{Sizes: sizes, Layers: layers}
}

func (mlp *MLP) Forward(x []*Value) []*Value {
	for _, l := range mlp.Layers {
		x = l.Forward(x)
	}

	return x
}

// Display

func (root *Value) DisplayGraph() {
	queue := []*Value{root}
	newQueue := []*Value{}

	for len(queue) != 0 {
		for _, v := range queue {
			v.Display()
			for _, child := range v.prev {
				newQueue = append(newQueue, child)
			}
		}
		queue = newQueue
		newQueue = []*Value{}
		fmt.Println("-----------------------------------------------------------")
	}

}

func (v *Value) Display() {
	prevVal := []float64{}
	for _, pv := range v.prev {
		prevVal = append(prevVal, pv.Data)
	}
	if v.op == "" && len(prevVal) == 0 {
		fmt.Printf("data %6.4f | grad %6.4f\n", v.Data, v.Grad)
	} else {
		fmt.Printf("data %6.4f | grad %6.4f | prev { %v %v }\n", v.Data, v.Grad, v.op, prevVal)
	}
}
