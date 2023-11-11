package main

import (
	"math/rand"
)

// Neuron

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

// Layer

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

// MLP

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

// Loss

func MSE(yPred, y []*Value) *Value {
	if len(y) != len(yPred) {
		panic("y and yPred different sizes")
	}
	sum := New(0)
	for i := 0; i < len(y); i++ {
		sum = Add(sum, Pow(Sub(yPred[i], y[i]), New(2)))
	}

	return sum
}
