package main

func main() {
	example1()
}

func example1() {
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

	// Tanh
	o := Tanh(n)
	// e := Exp(Mul(New(2), n))
	// o := Div(Sub(e, New(1)), Add(e, New(1)))

	o.Backward()

	o.DisplayGraph()
}

func example2() {
	x := []*Value{New(2), New(3), New(-1)}
	n := NewMLP(3, []int{4, 4, 1})
	out := n.Forward(x)
	out[0].Backward()

	out[0].DisplayGraph()
}
