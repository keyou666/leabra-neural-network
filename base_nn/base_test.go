package main

import (
	"fmt"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etable"
	"github.com/emer/leabra/leabra"
	"leabra_learning/data"
	"leabra_learning/model"
	"testing"
)

func TestModelBuild(t *testing.T) {
	hp := model.NewHipParams("BigHip")
	net := model.BuildModel(hp)

	dataMap := data.BuildData(0.1, hp, data.NewPatParams(20))

	trainEnv := &env.FixedTable{}
	trainEnv.Nm = "TrainEnv"
	trainEnv.Dsc = "training params and state"
	trainEnv.Table = etable.NewIdxView(dataMap["PreTest"])
	trainEnv.Validate()
	trainEnv.Run.Max = 30
	trainEnv.Init(0)

	lays := []string{"Input", "ECout"}
	trainEnv.Step()
	trainEnv.Counter(env.Epoch)
	net.InitExt()
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := trainEnv.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
	Out := net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	nn := Out.Shape().Len()
	targi, _ := Out.UnitVarIdx("Targ")
	//actMi, _ := Out.UnitVarIdx("ActM")
	for i := 0; i < nn; i++ {
		if i%49 == 0 {
			fmt.Println()
			fmt.Println(i/49, ":")
		}
		if i%7 == 0 {
			fmt.Println()
		}
		trg := Out.UnitVal1D(targi, i)
		//actm := Out.UnitVal1D(actMi, i)
		fmt.Print(trg, " ")
	}
}
