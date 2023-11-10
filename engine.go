package main

import (
	"fmt"
	"math"
)

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
