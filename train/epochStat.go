package train

import (
	"fmt"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"log"
)

func EpochStat(TrlLog *etable.Table, EpochLog *etable.Table, BaseInfo string, Run, Epoch int64, ListSize string, train bool) {
	// 筛选符合条件的行
	idx := etable.NewIdxView(TrlLog) // 创建视图

	// 筛选 Run 为 Run，Epoch 为 Epoch 的行
	idx.Filter(func(et *etable.Table, row int) bool {
		return et.CellString("BaseInfo", row) == BaseInfo && et.CellFloat("Run", row) == float64(Run) && et.CellFloat("Epoch", row) == float64(Epoch)
	})
	row := EpochLog.Rows
	EpochLog.SetNumRows(row + 1)

	if train {
		log.Printf("\n	BaseInfo %v,Run %v,Epoch %v\n	SSE %v \n"+
			"	AvgSSE %v\n"+
			"	PctErr %v\n"+
			"	PctCor %v\n"+
			"	CosDiff %v\n"+
			"	Mem %v\n"+
			"	TrgOnWasOff %v\n"+
			"	TrgOffWasOn %v\n"+
			"	AvgErrorSignal %v\n", BaseInfo, Run, Epoch, agg.Mean(idx, "SSE")[0], agg.Mean(idx, "AvgSSE")[0], agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
			return val > 0
		})[0], agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
			return val == 0
		})[0], agg.Mean(idx, "CosDiff")[0], agg.Mean(idx, "Mem")[0], agg.Mean(idx, "TrgOnWasOff")[0], agg.Mean(idx, "TrgOffWasOn")[0], agg.Mean(idx, "ErrorSignal")[0])

	} else {
		log.Printf("\n	BaseInfo %v,Run %v,Epoch %v\n	SSE %v \n"+
			"AvgSSE %v\n"+
			"PctErr %v\n"+
			"PctCor %v\n"+
			"CosDiff %v\n"+
			"ABMem %v\n"+
			"PctCor %v\n", BaseInfo, Run, Epoch, agg.Mean(idx, "SSE")[0], agg.Mean(idx, "AvgSSE")[0], agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
			return val > 0
		})[0], agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
			return val == 0
		})[0], agg.Mean(idx, "CosDiff")[0], agg.Mean(idx, "Mem")[0], agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
			return val == 0
		})[0])

	}

	EpochLog.SetCellString("BaseInfo", row, BaseInfo)
	EpochLog.SetCellString("NetSize", row, "default")
	EpochLog.SetCellString("ListSize", row, ListSize)
	EpochLog.SetCellFloat("Run", row, float64(Run))
	EpochLog.SetCellFloat("Epoch", row, float64(Epoch))
	EpochLog.SetCellFloat("SSE", row, agg.Mean(idx, "SSE")[0])
	EpochLog.SetCellFloat("AvgSSE", row, agg.Mean(idx, "AvgSSE")[0])
	EpochLog.SetCellFloat("PctErr", row, agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	EpochLog.SetCellFloat("PctCor", row, agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	EpochLog.SetCellFloat("CosDiff", row, agg.Mean(idx, "CosDiff")[0])

	if train {
		EpochLog.SetCellFloat("Mem", row, agg.Mean(idx, "Mem")[0])
		EpochLog.SetCellFloat("TrgOnWasOff", row, agg.Mean(idx, "TrgOnWasOff")[0])
		EpochLog.SetCellFloat("TrgOffWasOn", row, agg.Mean(idx, "TrgOffWasOn")[0])
		EpochLog.SetCellFloat("AvgErrorSignal", row, agg.Mean(idx, "ErrorSignal")[0])
	} else {
		EpochLog.SetCellFloat("ABMem", row, agg.Mean(idx, "Mem")[0])
		EpochLog.SetCellFloat("PctCor", row, agg.PropIf(idx, "SSE", func(idx int, val float64) bool {
			return val == 0
		})[0])
	}

}

func EpochTableToStringLastLine(EpochLog *etable.Table) string {
	row := EpochLog.Rows - 1

	str := "\n"
	for _, nm := range EpochLog.ColNames {
		strValue, err := EpochLog.CellStringTry(nm, row)
		if err == nil {
			str += fmt.Sprintf("	%v : %v\n", nm, strValue)
			continue
		}
		floatValue, err := EpochLog.CellFloatTry(nm, row)
		if err == nil {
			str += fmt.Sprintf("	%v : %v\n", nm, floatValue)
			continue
		}
	}
	return str

}
