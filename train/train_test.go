package train

import (
	"fmt"
	"github.com/emer/leabra/leabra"
	"leabra_learning/data"
	"leabra_learning/model"
	"testing"
)

func TestTrainRre(t *testing.T) {
	sim := Sim{
		TrnTrlLog: CreateTrainTrailLogTable(),
		TstTrlLog: CreateTestTrailLogTable(),
		TrnEpcLog: CreateTrainEpochLogTable(),
		TstEpcLog: CreateTestEpochLogTable(),
		RunLog:    CreateRunLogTable(),
		Time: &leabra.Time{
			CycPerQtr: 25,
		},
		MemThr: 0.34,
	}
	pa := data.NewPatParams(60)
	hp := model.NewHipParams("BigHip")
	sim.Hp = hp
	sim.Pa = pa
	sim.DataMap = data.BuildData(draft, hp, pa)

	sim.EDL = true
	sim.Net = model.BuildModel(hp)
	type_ := "PreTranInit"
	baseInfo := type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	//sim.PreTrainInit(baseInfo, "PreTrainInit")

	sim.EDL = true
	//sim.Net = model.BuildModel(hp)
	type_ = "PreTest"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.TrainRP(baseInfo, "PreTest")

	type_ = "PreTrain"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.Train(baseInfo, "PreTrain")

	type_ = "PreTestAB"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.Test(baseInfo, "TestAB")

}

func TestTrainPost(t *testing.T) {
	sim := Sim{
		TrnTrlLog: CreateTrainTrailLogTable(),
		TstTrlLog: CreateTestTrailLogTable(),
		TrnEpcLog: CreateTrainEpochLogTable(),
		TstEpcLog: CreateTestEpochLogTable(),
		RunLog:    CreateRunLogTable(),
		Time: &leabra.Time{
			CycPerQtr: 25,
		},
		MemThr: 0.34,
	}
	pa := data.NewPatParams(60)
	hp := model.NewHipParams("BigHip")
	sim.Hp = hp
	sim.Pa = pa
	sim.DataMap = data.BuildData(draft, hp, pa)

	sim.EDL = true
	sim.Net = model.BuildModel(hp)
	type_ := "PreTrainInit"
	baseInfo := type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	//sim.PreTrainInit(baseInfo, "PreTrainInit")

	sim.EDL = true
	//sim.Net = model.BuildModel(hp)
	type_ = "PostTrain"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.Train(baseInfo, "PostTrain")

	type_ = "PostTest"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.TrainRP(baseInfo, "PostTest")

	type_ = "PostTestAB"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.Test(baseInfo, "TestAB")

}

func TestTrainPost2(t *testing.T) {
	sim := Sim2{
		TrnTrlLog: CreateTrainTrailLogTable(),
		TstTrlLog: CreateTestTrailLogTable(),
		TrnEpcLog: CreateTrainEpochLogTable(),
		TstEpcLog: CreateTestEpochLogTable(),
		RunLog:    CreateRunLogTable(),
		Time: &leabra.Time{
			CycPerQtr: 25,
		},
		MemThr: 0.34,
	}
	pa := data.NewPatParams(60)
	hp := model.NewHipParams("BigHip")
	sim.Hp = hp
	sim.Pa = pa
	sim.DataMap = data.BuildData(draft, hp, pa)

	sim.EDL = true
	sim.Net = model.BuildModel(hp)
	type_ := "PreTrainInit"
	baseInfo := type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	//sim.PreTrainInit(baseInfo, "PreTrainInit")

	sim.EDL = true
	//sim.Net = model.BuildModel(hp)
	type_ = "PostTrain"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.Train(baseInfo, "PostTrain")

	type_ = "PostTest"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.TrainRP(baseInfo, "PostTest")

	type_ = "PostTestAB"
	baseInfo = type_ + "_" + "EDL" + "_" + "ListSize" + fmt.Sprint(60) + "_" + "BigHip"
	sim.Test(baseInfo, "TestAB")

}
