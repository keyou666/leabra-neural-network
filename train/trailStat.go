package train

import (
	"fmt"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/metric"
	"github.com/emer/leabra/leabra"
	"math"
)

type TrailBaseInfo struct {
	BaseInfo    string
	Run         int64
	Epoch       int64
	Trail       int64
	TrailName   string
	SSE         float64
	AvgSSE      float64
	CosDiff     float64
	Mem         float64
	TrgOnWasOff float64
	TrgOffWasOn float64
	ErrorSignal float64
}

func (t *TrailBaseInfo) MemStat(net *leabra.Network, MemThr float64, train bool) {
	ecout := net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ecin := net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	nn := ecout.Shape().Len()
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIdx("ActM")
	targi, _ := ecout.UnitVarIdx("Targ")
	actQ1i, _ := ecout.UnitVarIdx("ActQ1")
	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitVal1D(actMi, ni)
		trg := ecout.UnitVal1D(targi, ni) // full pattern target
		inact := ecin.UnitVal1D(actQ1i, ni)

		if trg < 0.5 { // trgOff
			trgOffN += 1
			if actm > 0.5 {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < 0.5 { // missing in ECin -- completion target
				cmpN += 1
				if actm < 0.5 {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < 0.5 {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	var Mem float64
	if train {
		if trgOnWasOffAll < MemThr && trgOffWasOn < MemThr {
			Mem = 1
		} else {
			Mem = 0
		}
	} else {
		if cmpN > 0 {
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < MemThr && trgOffWasOn < MemThr {
				Mem = 1
			} else {
				Mem = 0
			}
		}
	}

	t.Mem = Mem

	if train {
		t.TrgOnWasOff = trgOnWasOffAll
	} else {
		t.TrgOnWasOff = trgOnWasOffCmp
	}
	t.TrgOffWasOn = trgOffWasOn

}

func (t *TrailBaseInfo) ErrorSignalStat(net *leabra.Network) {
	Out := net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	nn := Out.Shape().Len()
	targi, _ := Out.UnitVarIdx("Targ")
	actMi, _ := Out.UnitVarIdx("ActM")

	active := make([]float32, 0)
	target := make([]float32, 0)

	for i := 0; i < nn; i++ {
		// 只计算B
		if i/49 == 1 {
			trg := Out.UnitVal1D(targi, i)
			actm := Out.UnitVal1D(actMi, i)
			active = append(active, actm)
			target = append(target, trg)
		}
	}

	t.ErrorSignal = math.Abs(float64(metric.Correlation32(active, target)))
}

func (t *TrailBaseInfo) SSEStat(net *leabra.Network) {
	outLay := net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	t.CosDiff = float64(outLay.CosDiff.Cos)
	t.SSE, t.AvgSSE = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5

	return
}

func (t *TrailBaseInfo) BaseStat(BaseInfo string, Run int64, Epoch int64, Trail int64, TrailName string) {
	t.BaseInfo = BaseInfo
	t.Run = Run
	t.Epoch = Epoch
	t.Trail = Trail
	t.TrailName = TrailName
}

func (t *TrailBaseInfo) SetToTrlLog(TrlLog *etable.Table) {
	row := TrlLog.Rows

	TrlLog.SetNumRows(row + 1)
	TrlLog.SetCellString("BaseInfo", row, t.BaseInfo)
	TrlLog.SetCellFloat("Run", row, float64(t.Run))
	TrlLog.SetCellFloat("Epoch", row, float64(t.Epoch))
	TrlLog.SetCellFloat("Trial", row, float64(t.Trail))
	TrlLog.SetCellString("TrialName", row, t.TrailName)
	TrlLog.SetCellFloat("SSE", row, t.SSE)
	TrlLog.SetCellFloat("AvgSSE", row, t.AvgSSE)
	TrlLog.SetCellFloat("CosDiff", row, t.CosDiff)

	TrlLog.SetCellFloat("Mem", row, t.Mem)
	TrlLog.SetCellFloat("TrgOnWasOff", row, t.TrgOnWasOff)
	TrlLog.SetCellFloat("TrgOffWasOn", row, t.TrgOffWasOn)
	TrlLog.SetCellFloat("ErrorSignal", row, t.ErrorSignal)

}

func (t *TrailBaseInfo) ToString() string {

	return fmt.Sprintf("\n	BaseInfo : %v\n"+
		"	Run : %v\n"+
		"	Epoch : %v\n"+
		"	Trail : %v\n"+
		"	TrailName : %v\n"+
		"	SSE : %v\n"+
		"	AvgSSE : %v\n"+
		"	CosDiff : %v\n"+
		"	Mem : %v\n"+
		"	TrgOnWasOff : %v\n"+
		"	TrgOffWasOn : %v\n"+
		"	ErrorSignal : %v\n", t.BaseInfo, t.Run, t.Epoch, t.Trail, t.TrailName, t.SSE, t.AvgSSE, t.CosDiff, t.Mem, t.TrgOnWasOff, t.TrgOffWasOn, t.ErrorSignal)
}
