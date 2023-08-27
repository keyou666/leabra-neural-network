package model

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/leabra/hip"
	"github.com/emer/leabra/leabra"
	"log"
)

var ParamSetsNet = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "keeping default params for generic prjns",
				Params: params.Params{
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.WtBal.On":    "false",
				}},
			{Sel: ".EcCa1Prjn", Desc: "encoder projections -- no norm, moment",
				Params: params.Params{
					"Prjn.Learn.Lrate":       "0.04",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true", // counteracting hogging
					//"Prjn.Learn.XCal.SetLLrn": "true", // bcm now avail, comment out = default LLrn
					//"Prjn.Learn.XCal.LLrn":    "0",    // 0 = turn off BCM, must with SetLLrn = true
				}},
			{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
				Params: params.Params{
					"Prjn.CHL.Hebb":          "0.01", // .01 > .05? > .1?
					"Prjn.Learn.Lrate":       "0.2",  // .2 probably better? .4 was prev default
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: ".PPath", Desc: "performant path, new Dg error-driven EcCa1Prjn prjns",
				Params: params.Params{
					"Prjn.Learn.Lrate":       "0.15", // err driven: .15 > .2 > .25 > .1
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
					//"Prjn.Learn.XCal.SetLLrn": "true", // bcm now avail, comment out = default LLrn
					//"Prjn.Learn.XCal.LLrn":    "0",    // 0 = turn off BCM, must with SetLLrn = true
				}},
			{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
				Params: params.Params{
					"Prjn.WtScale.Abs": "4.0", // 4 > 6 > 2 (fails)
				}},
			{Sel: "#InputToECin", Desc: "one-to-one input to EC",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0.0",
				}},
			{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.9",
					"Prjn.WtInit.Var":  "0.01",
					"Prjn.WtScale.Rel": "0.5", // .5 = .3? > .8 (fails); zycyc test this
				}},
			{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false", // learning here definitely does NOT work!
					"Prjn.WtInit.Mean": "0.9",
					"Prjn.WtInit.Var":  "0.01",
					"Prjn.WtScale.Rel": "4", // err del 4: 4 > 6 > 8
					//"Prjn.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			//{Sel: "#ECinToCA3", Desc: "ECin Perforant Path",
			//	Params: params.Params{
			//		"Prjn.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
			//	}},
			{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons: rel=2 still the best",
				Params: params.Params{
					"Prjn.WtScale.Rel": "2",   // 2 > 1 > .5 = .1
					"Prjn.Learn.Lrate": "0.1", // .1 > .08 (close) > .15 > .2 > .04;
					//"Prjn.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn":       "true", // absolutely essential to have on! learning slow if off.
					"Prjn.CHL.Hebb":          "0.2",  // .2 seems good
					"Prjn.CHL.SAvgCor":       "0.1",  // 0.01 = 0.05 = .1 > .2 > .3 > .4 (listlize 20-100)
					"Prjn.CHL.MinusQ1":       "true", // dg self err slightly better
					"Prjn.Learn.Lrate":       "0.05", // .05 > .1 > .2 > .4; .01 less interference more learning time - key tradeoff param, .05 best for list20-100
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					"Prjn.CHL.Hebb":          "0.01", // .01 > .005 > .02 > .002 > .001 > .05 (crazy)
					"Prjn.CHL.SAvgCor":       "0.4",
					"Prjn.Learn.Lrate":       "0.1", // CHL: .1 =~ .08 > .15 > .2, .05 (sig worse)
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
					//"Prjn.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			//{Sel: "#ECinToCA1", Desc: "ECin Perforant Path",
			//	Params: params.Params{
			//		"Prjn.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
			//	}},
			//{Sel: "#ECoutToCA1", Desc: "ECout Perforant Path",
			//	Params: params.Params{
			//		"Prjn.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
			//	}},
			{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
				Params: params.Params{
					"Layer.Act.Gbar.L":        "0.1",
					"Layer.Inhib.ActAvg.Init": "0.2",
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.Pool.Gi":     "2.0",
					"Layer.Inhib.Pool.On":     "true",
				}},
			{Sel: "#DG", Desc: "very sparse = high inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.01",
					"Layer.Inhib.Layer.Gi":    "3.8", // 3.8 > 3.6 > 4.0 (too far -- tanks)
				}},
			{Sel: "#CA3", Desc: "sparse = high inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Layer.Gi":    "2.8", // 2.8 = 3.0 really -- some better, some worse
					"Layer.Learn.AvgL.Gain":   "2.5", // stick with 2.5
				}},
			{Sel: "#CA1", Desc: "CA1 only Pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "2.4", // 2.4 > 2.2 > 2.6 > 2.8 -- 2.4 better *for small net* but not for larger!
					"Layer.Learn.AvgL.Gain":   "2.5", // 2.5 > 2 > 3
					//"Layer.Inhib.ActAvg.UseFirst": "false", // first activity is too low, throws off scaling, from Randy, zycyc: do we need this?
				}},
		},
		// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
		// to initialize the network every time, even if it is a different size..
	}},
}

var ParamSets = params.Sets{
	{Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "10",
					"HipParams.CA1Pool.X": "10",
					"HipParams.CA3Size.Y": "20",
					"HipParams.CA3Size.X": "20",
					"HipParams.DGRatio":   "2.236", // 1.5 before, sqrt(5) aligns with Ketz et al. 2013
				}},
		},
	}},
	{Name: "MedHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "15",
					"HipParams.CA1Pool.X": "15",
					"HipParams.CA3Size.Y": "30",
					"HipParams.CA3Size.X": "30",
					"HipParams.DGRatio":   "2.236", // 1.5 before
				}},
		},
	}},
	{Name: "BigHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "20",
					"HipParams.CA1Pool.X": "20",
					"HipParams.CA3Size.Y": "40",
					"HipParams.CA3Size.X": "40",
					"HipParams.DGRatio":   "2.236", // 1.5 before
				}},
		},
	}}}

type HipParams struct {
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
}

func NewHipParams(Hip string) *HipParams {
	hp := &HipParams{}
	// size
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(15, 15) // using MedHip for default
	hp.CA3Size.Set(30, 30) // using MedHip for default
	hp.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// ratio
	hp.DGPCon = 0.25
	hp.CA3PCon = 0.25
	hp.MossyPCon = 0.02
	hp.ECPctAct = 0.2

	hp.MossyDel = 4     // 4 > 2 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 3 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount..

	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)

	if Hip != "SmallHip" && Hip != "MedHip" && Hip != "BigHip" {
		log.Printf("Hip should be SmallHip,MedHip or BigHip.\n")
		return nil
	}

	pset, err := ParamSets.SetByNameTry(Hip)
	if err != nil {
		log.Printf("ParamSets parse error : %v/n", err)
		return nil
	}

	simp, ok := pset.Sheets["Hip"]
	if ok {
		simp.Apply(hp, true)
	}
	return hp
}

func BuildModel(hp *HipParams) *leabra.Network {

	net := &leabra.Network{}
	net.InitName(net, "model")

	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Input)
	ecin := net.AddLayer4D("ECin", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Hidden)
	ecout := net.AddLayer4D("ECout", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Target) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, emer.Hidden)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, emer.Hidden)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(in, ecin, onetoone, emer.Forward)
	net.ConnectLayers(ecout, ecin, onetoone, emer.Back)

	pj := net.ConnectLayersPrjn(ecin, ca1, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ca1, ecout, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ecout, ca1, pool1to1, emer.Back, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")

	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon

	pj = net.ConnectLayersPrjn(ecin, dg, ppathDG, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("PPath")
	pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.EcCa1Prjn{})
	pj.SetClass("PPath")

	pj = net.ConnectLayersPrjn(ca3, ca1, full, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	mossy := prjn.NewUnifRnd()
	mossy.PCon = hp.MossyPCon
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, &hip.CHLPrjn{}) // no learning
	pj.SetClass("HippoCHL")

	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3)

	net.Defaults()
	pset, _ := ParamSetsNet.SetByNameTry("Base")
	netp, _ := pset.Sheets["Network"]
	net.ApplyParams(netp, true)

	err := net.Build()
	if err != nil {
		log.Printf("build model error : %v\n", err)
		return nil
	}
	net.InitWts()

	return net
}
