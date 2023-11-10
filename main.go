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
	o := Tanh(n, "o")

	o.DisplayGraph()
}

// Value struct

type Value struct {
	Data  float64
	Grad  float64
	prev  []*Value
	op    string
	Label string
}

func New(f float64, l string) *Value {
	return &Value{Data: f, Label: l}

}

// Operators

func Add(a, b *Value, l string) *Value {
	return &Value{
		Data:  a.Data + b.Data,
		prev:  []*Value{a, b},
		op:    "+",
		Label: l,
	}
}

func Mul(a, b *Value, l string) *Value {
	return &Value{
		Data:  a.Data * b.Data,
		prev:  []*Value{a, b},
		op:    "*",
		Label: l,
	}
}

func Tanh(x *Value, l string) *Value {
	return &Value{
		Data:  (math.Exp(2*x.Data) - 1) / (math.Exp(2*x.Data) + 1),
		prev:  []*Value{x},
		op:    "tanh",
		Label: l,
	}
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
		fmt.Println("-------------------------------------------------")
	}

}

func (v *Value) Display() {
	prevVal := []string{}
	for _, pv := range v.prev {
		prevVal = append(prevVal, pv.Label)
	}
	if v.op == "" && len(prevVal) == 0 {
		fmt.Printf("%v | data %5.2f | grad %5.2f\n", v.Label, v.Data, v.Grad)
	} else {
		fmt.Printf("%v | data %5.2f | grad %5.2f | prev { %v %v }\n", v.Label, v.Data, v.Grad, v.op, prevVal)
	}
}
