package main

import "fmt"

func main() {
	a := New(2.0, "a")
	b := New(-3.0, "b")
	c := New(10.0, "c")
	e := Mul(a, b, "e")
	d := Add(e, c, "d")
	f := New(-2.0, "c")
	L := Mul(d, f, "L")

	L.DisplayGraph()
}

type Value struct {
	Data  float64
	prev  []*Value
	op    string
	label string
}

func New(f float64, l string) *Value {
	return &Value{Data: f, label: l}

}

func Add(a, b *Value, l string) *Value {
	return &Value{
		Data:  a.Data + b.Data,
		prev:  []*Value{a, b},
		op:    "+",
		label: l,
	}
}

func Mul(a, b *Value, l string) *Value {
	return &Value{
		Data:  a.Data * b.Data,
		prev:  []*Value{a, b},
		op:    "*",
		label: l,
	}
}

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
		fmt.Println("-------------------")
	}

}

func (v *Value) Display() {
	prevVal := []string{}
	for _, pv := range v.prev {
		prevVal = append(prevVal, pv.label)
	}
	if v.op == "" && len(prevVal) == 0 {
		fmt.Printf("%v: %v\n", v.label, v.Data)
	} else {
		fmt.Printf("%v: %v { %v %v }\n", v.label, v.Data, v.op, prevVal)
	}
}
