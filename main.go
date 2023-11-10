package main

import (
	"fmt"
	"math"
)

func main() {
	// inputs x1, x2
	x1 := New(2.0, "x1")
	x2 := New(0.0, "x2")

	// weights w1, w2
	w1 := New(-3.0, "w1")
	w2 := New(1.0, "w2")

	// bias
	b := New(6.8813735870195432, "b")

	// x1*w1 + x2*w2 + b
	w1x1 := Mul(w1, x1, "w1x1")
	w2x2 := Mul(w2, x2, "w2x2")
	w1x1w2x2 := Add(w1x1, w2x2, "w1x1w2x2")
	n := Add(w1x1w2x2, b, "n")

	// Tanh
	o := Tanh(n, "o")
	// e := Exp(Mul(New(2, ""), n, ""), "e")
	// o := Div(Sub(e, New(1, ""), ""), Add(e, New(1, ""), ""), "o")

	o.Backward()

	o.DisplayGraph()
}

// Value struct

type Value struct {
	Data     float64
	Grad     float64
	backward func()
	prev     []*Value
	op       string
	Label    string
}

func New(f float64, l string) *Value {
	return &Value{Data: f, Label: l}
}

// Operators

func Add(a, b *Value, l string) *Value {
	out := &Value{
		Data:  a.Data + b.Data,
		prev:  []*Value{a, b},
		op:    "+",
		Label: l,
	}
	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}

	return out
}

func Neg(x *Value, l string) *Value {
	return Mul(x, New(-1, ""), l)
}

func Sub(a, b *Value, l string) *Value {
	return Add(a, Neg(b, ""), l)
}

func Mul(a, b *Value, l string) *Value {
	out := &Value{
		Data:  a.Data * b.Data,
		prev:  []*Value{a, b},
		op:    "*",
		Label: l,
	}
	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}

	return out
}

func Div(a, b *Value, l string) *Value {
	return Mul(a, Pow(b, New(-1, ""), ""), l)
}

func Pow(a, b *Value, l string) *Value {
	out := &Value{
		Data:  math.Pow(a.Data, b.Data),
		prev:  []*Value{a, b},
		op:    "Pow",
		Label: l,
	}
	out.backward = func() {
		a.Grad += (b.Data * math.Pow(a.Data, b.Data-1)) * out.Grad
		b.Grad += (a.Data * math.Pow(b.Data, a.Data-1)) * out.Grad
	}

	return out
}

func Tanh(x *Value, l string) *Value {
	out := &Value{
		Data:  (math.Exp(2*x.Data) - 1) / (math.Exp(2*x.Data) + 1),
		prev:  []*Value{x},
		op:    "tanh",
		Label: l,
	}
	out.backward = func() {
		x.Grad += (1 - math.Pow(out.Data, 2)) * out.Grad
	}

	return out
}

func Exp(x *Value, l string) *Value {
	out := &Value{
		Data:  math.Exp(x.Data),
		prev:  []*Value{x},
		op:    "exp",
		Label: l,
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
	prevVal := []string{}
	for _, pv := range v.prev {
		prevVal = append(prevVal, pv.Label)
	}
	if v.op == "" && len(prevVal) == 0 {
		fmt.Printf("%v | data %6.4f | grad %6.4f\n", v.Label, v.Data, v.Grad)
	} else {
		fmt.Printf("%v | data %6.4f | grad %6.4f | prev { %v %v }\n", v.Label, v.Data, v.Grad, v.op, prevVal)
	}
}
