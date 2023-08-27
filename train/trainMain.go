package train

import (
	"fmt"
	"github.com/emer/etable/etable"
	"github.com/emer/leabra/leabra"
	"leabra_learning/data"
	"leabra_learning/model"
	"log"
	"os"
)

const draft = 0.1

var ListSize = []int{20, 40, 60, 80, 100}
var Hip = []string{"SmallHip", "MedHip", "BigHip"}
var Type = []string{"Pre", "Post"}
var EDL = []bool{true, false}

const TrnTrailLogPath = "TrnTrailLog.csv"
const TrnEpochLogPath = "TrnEpochLog.csv"
const TstTrailLogPath = "TstTrailLog.csv"
const TstEpochLogPath = "TstEpochLog.csv"

func Start() {

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

	for _, listSize := range ListSize {
		for _, hip := range Hip {

			pa := data.NewPatParams(listSize)
			hp := model.NewHipParams(hip)
			sim.Hp = hp
			sim.Pa = pa
			sim.DataMap = data.BuildData(draft, hp, pa)

			for _, edl := range EDL {
				sim.EDL = edl
				edlStr := "NoEDL"
				if edl {
					edlStr = "EDL"
				}
				for _, type_ := range Type {

					sim.Net = model.BuildModel(hp)

					if type_ == "Pre" {

						baseInfo := type_ + "Test" + "_" + edlStr + "_" + "ListSize" + fmt.Sprint(listSize) + "_" + hip
						sim.TrainRP(baseInfo, "PreTest")
						baseInfo = type_ + "Train" + "_" + edlStr + "_" + "ListSize" + fmt.Sprint(listSize) + "_" + hip
						sim.Train(baseInfo, "PreTrain")
						baseInfo = type_ + "TestAB" + "_" + edlStr + "_" + "ListSize" + fmt.Sprint(listSize) + "_" + hip
						sim.Test(baseInfo, "TestAB")

					} else if type_ == "Post" {

						baseInfo := type_ + "Train" + "_" + edlStr + "_" + "ListSize" + fmt.Sprint(listSize) + "_" + hip
						sim.Train(baseInfo, "PostTrain")
						baseInfo = type_ + "Test" + "_" + edlStr + "_" + "ListSize" + fmt.Sprint(listSize) + "_" + hip
						sim.TrainRP(baseInfo, "PostTest")
						baseInfo = type_ + "TestAB" + "_" + edlStr + "_" + "ListSize" + fmt.Sprint(listSize) + "_" + hip
						sim.Test(baseInfo, "TestAB")

					}

					WriteToCsv(sim.TrnTrlLog, TrnTrailLogPath)
					WriteToCsv(sim.TrnEpcLog, TrnEpochLogPath)
					WriteToCsv(sim.TstTrlLog, TstTrailLogPath)
					WriteToCsv(sim.TstEpcLog, TstEpochLogPath)
				}
			}
		}
	}
}

func WriteToCsv(dt *etable.Table, filePath string) {
	file, err := os.Create(filePath)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	defer file.Close()

	err = dt.WriteCSV(file, etable.Comma, true)
	if err != nil {
		log.Fatalf("Failed to WriteCSV file: %v", err)
	}

	// 确保所有内容都已写入文件
	err = file.Sync()
	if err != nil {
		log.Fatalf("Failed to sync file: %v", err)
	}

	// 关闭文件
	err = file.Close()
	if err != nil {
		log.Fatalf("Failed to close file: %v", err)
	}
}
