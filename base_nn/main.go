package main

import (
	"fmt"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"log"
)

func main() {

	net := &leabra.Network{}
	net.InitName(net, "base_nn")
	inp := net.AddLayer2D("Input", 2, 1, emer.Input)
	hid := net.AddLayer2D("Hidden", 4, 4, emer.Hidden)
	out := net.AddLayer2D("Output", 2, 1, emer.Target)

	full := prjn.NewFull()
	net.ConnectLayers(inp, hid, full, emer.Forward)
	net.ConnectLayers(hid, out, full, emer.Forward)
	net.Defaults()
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()

	trainX := etensor.NewFloat32([]int{4, 2, 1}, nil, []string{"sample_number", "X", "Y"})
	trainX.SetFloats([]float64{0, 0, 0, 1, 1, 0, 1, 1})

	trainY := etensor.NewFloat32([]int{4, 2, 1}, nil, []string{"sample_number", "X", "Y"})
	trainY.SetFloats([]float64{0, 0, 1, 1, 1, 1, 1, 1})

	data := make(patgen.Vocab)
	data["trainX"] = trainX
	data["trainY"] = trainY

	trainTable := &etable.Table{}
	patgen.InitPats(trainTable, "train", "", "Input", "Output", 4, 1, 1, 2, 1)
	patgen.MixPats(trainTable, data, "Input", []string{"trainX"})
	patgen.MixPats(trainTable, data, "Output", []string{"trainY"})

	trainEnv := &env.FixedTable{}
	trainEnv.Nm = "TrainEnv"
	trainEnv.Dsc = "training params and state"
	trainEnv.Table = etable.NewIdxView(trainTable)
	trainEnv.Validate()
	trainEnv.Run.Max = 30
	trainEnv.Init(0)

	Out := net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	Out.SetType(emer.Target) // clamp a plus phase during testing
	Out.UpdateExtFlags()

	lays := []string{"Input", "Output"}
	for i := 0; i < 1000; i++ {
		trainEnv.Step()
		epc, _, chg := trainEnv.Counter(env.Epoch)
		fmt.Println(epc, chg)
		net.InitExt()
		for _, lnm := range lays {
			ly := net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
			pats := trainEnv.State(ly.Nm)
			if pats != nil {
				ly.ApplyExt(pats)
			}
		}

		net.AlphaCycInit(true)
		trainTime := &leabra.Time{
			CycPerQtr: 25,
		}
		trainTime.AlphaCycStart()
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < trainTime.CycPerQtr; cyc++ {
				net.Cycle(trainTime)
				trainTime.CycleInc()
			}
			net.QuarterFinal(trainTime)

			if qtr+1 == 3 {
				Out = net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
				actMi, _ := Out.UnitVarIdx("ActM")
				actm := Out.UnitVal1D(actMi, 0)
				targi, _ := Out.UnitVarIdx("Targ")
				trg := Out.UnitVal1D(targi, 0)
				fmt.Println(qtr, actm, trg)
				Out = net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
				actMi, _ = Out.UnitVarIdx("ActM")
				actm = Out.UnitVal1D(actMi, 1)
				targi, _ = Out.UnitVarIdx("Targ")
				trg = Out.UnitVal1D(targi, 1)
				fmt.Println(qtr, actm, trg)
			}
			trainTime.QuarterInc()
		}
		net.DWt()
		net.WtFmDWt()

	}

}
