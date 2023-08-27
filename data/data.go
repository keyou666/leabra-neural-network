package data

import (
	"fmt"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etable"
	"leabra_learning/model"
	"math/rand"
)

type PatParams struct {
	ListSize    int     `desc:"number of A-B, A-C patterns each"`
	MinDiffPct  float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
	DriftCtxt   bool    `desc:"use drifting context representations -- otherwise does bit flips from prototype"`
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`
}

func NewPatParams(listSize int) *PatParams {
	pa := &PatParams{
		ListSize:    100,
		MinDiffPct:  0.5,
		CtxtFlipPct: .25,
	}
	pa.ListSize = listSize
	return pa
}

func BuildData(drate float32, hp *model.HipParams, pa *PatParams) map[string]*etable.Table {
	rand.Seed(0)
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	npats := pa.ListSize
	pctAct := hp.ECPctAct
	minDiff := pa.MinDiffPct
	nOn := patgen.NFmPct(pctAct, plY*plX)
	ctxtflip := patgen.NFmPct(pa.CtxtFlipPct, nOn)
	drift := patgen.NFmPct(drate, nOn)

	VocabMap := make(patgen.Vocab)
	patgen.AddVocabEmpty(VocabMap, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(VocabMap, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(VocabMap, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(VocabMap, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(VocabMap, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(VocabMap, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(VocabMap, "ctxt", 3, plY, plX, pctAct, minDiff)

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(VocabMap, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
	}

	patgen.AddVocabClone(VocabMap, "ctxt1s1", "ctxt1")
	patgen.AddVocabClone(VocabMap, "ctxt2s1", "ctxt2")
	patgen.AddVocabClone(VocabMap, "ctxt3s1", "ctxt3")
	patgen.AddVocabClone(VocabMap, "ctxt4s1", "ctxt4")

	patgen.FlipBitsRows(VocabMap["ctxt1s1"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt2s1"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt3s1"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt4s1"], drift, drift, 1, 0)

	patgen.AddVocabClone(VocabMap, "ctxt1s2", "ctxt1s1")
	patgen.AddVocabClone(VocabMap, "ctxt2s2", "ctxt2s1")
	patgen.AddVocabClone(VocabMap, "ctxt3s2", "ctxt3s1")
	patgen.AddVocabClone(VocabMap, "ctxt4s2", "ctxt4s1")

	patgen.FlipBitsRows(VocabMap["ctxt1s2"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt2s2"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt3s2"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt4s2"], drift, drift, 1, 0)

	patgen.AddVocabClone(VocabMap, "ctxt1t", "ctxt1s2")
	patgen.AddVocabClone(VocabMap, "ctxt2t", "ctxt2s2")
	patgen.AddVocabClone(VocabMap, "ctxt3t", "ctxt3s2")
	patgen.AddVocabClone(VocabMap, "ctxt4t", "ctxt4s2")

	patgen.FlipBitsRows(VocabMap["ctxt1t"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt2t"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt3t"], drift, drift, 1, 0)
	patgen.FlipBitsRows(VocabMap["ctxt4t"], drift, drift, 1, 0)

	PreTrainInit := &etable.Table{}
	patgen.InitPats(PreTrainInit, "PreTrainInit", "PreTrainInit Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(PreTrainInit, VocabMap, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(PreTrainInit, VocabMap, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	PreTest := &etable.Table{}
	patgen.InitPats(PreTest, "PreTest", "PreTest Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(PreTest, VocabMap, "Input", []string{"A", "empty", "ctxt1s1", "ctxt2s1", "ctxt3s1", "ctxt4s1"})
	patgen.MixPats(PreTest, VocabMap, "ECout", []string{"A", "B", "ctxt1s1", "ctxt2s1", "ctxt3s1", "ctxt4s1"})

	PreTrain := &etable.Table{}
	patgen.InitPats(PreTrain, "PreTrain", "PreTrain Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(PreTrain, VocabMap, "Input", []string{"A", "B", "ctxt1s2", "ctxt2s2", "ctxt3s2", "ctxt4s2"})
	patgen.MixPats(PreTrain, VocabMap, "ECout", []string{"A", "B", "ctxt1s2", "ctxt2s2", "ctxt3s2", "ctxt4s2"})

	PostTrain := &etable.Table{}
	patgen.InitPats(PostTrain, "PostTrain", "PreTrain Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(PostTrain, VocabMap, "Input", []string{"A", "B", "ctxt1s1", "ctxt2s1", "ctxt3s1", "ctxt4s1"})
	patgen.MixPats(PostTrain, VocabMap, "ECout", []string{"A", "B", "ctxt1s1", "ctxt2s1", "ctxt3s1", "ctxt4s1"})

	PostTest := &etable.Table{}
	patgen.InitPats(PostTest, "PostTest", "PreTrain Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(PostTest, VocabMap, "Input", []string{"A", "empty", "ctxt1s2", "ctxt2s2", "ctxt3s2", "ctxt4s2"})
	patgen.MixPats(PostTest, VocabMap, "ECout", []string{"A", "B", "ctxt1s2", "ctxt2s2", "ctxt3s2", "ctxt4s2"})

	TestAB := &etable.Table{}
	patgen.InitPats(TestAB, "TestAB", "TestAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(TestAB, VocabMap, "Input", []string{"A", "empty", "ctxt1t", "ctxt2t", "ctxt3t", "ctxt4t"})
	patgen.MixPats(TestAB, VocabMap, "ECout", []string{"A", "B", "ctxt1t", "ctxt2t", "ctxt3t", "ctxt4t"})

	dataMap := make(map[string]*etable.Table)
	dataMap["PreTrainInit"] = PreTrainInit
	dataMap["PreTest"] = PreTest
	dataMap["PreTrain"] = PreTrain
	dataMap["PostTrain"] = PostTrain
	dataMap["PostTest"] = PostTest
	dataMap["TestAB"] = TestAB

	return dataMap

}
