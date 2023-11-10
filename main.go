package main

import "fmt"

func main() {
	a := New(2.0)
	b := New(-3.0)
	c := New(10.0)
	d := Add(Mul(a, b), c)

	d.DisplayGraph()
}

type Value struct {
	Data float64
	prev []*Value
	op   string
}

func New(f float64) *Value {
	return &Value{Data: f}

}

func Add(a, b *Value) *Value {
	return &Value{
		Data: a.Data + b.Data,
		prev: []*Value{a, b},
		op:   "+",
	}
}

func Mul(a, b *Value) *Value {
	return &Value{
		Data: a.Data * b.Data,
		prev: []*Value{a, b},
		op:   "*",
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
	prevVal := []float64{}
	for _, pv := range v.prev {
		prevVal = append(prevVal, pv.Data)
	}
	if v.op == "" && len(prevVal) == 0 {
		fmt.Printf("%v { }\n", v.Data)
	} else {
		fmt.Printf("%v { %v %v }\n", v.Data, v.op, prevVal)
	}
}
