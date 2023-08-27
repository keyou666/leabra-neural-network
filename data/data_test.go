package data

import (
	"fmt"
	"leabra_learning/model"
	"testing"
)

func TestBuildData(t *testing.T) {
	pa := NewPatParams(60)
	hp := model.NewHipParams("BigHip")

	DataMap := BuildData(0.1, hp, pa)
	_ = DataMap

	PostTrain := DataMap["PostTrain"]

	for i := 0; i < 49; i++ {
		if i%7 == 0 {
			fmt.Println()
		}

		fmt.Print(PostTrain.Col(1).FloatVal1D(i))
		fmt.Print(" ")

	}
}
