package train

import (
	"fmt"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"leabra_learning/data"
	"leabra_learning/model"
	"log"
)

type Sim struct {
	Net     *leabra.Network
	Hp      *model.HipParams
	Pa      *data.PatParams
	DataMap map[string]*etable.Table

	TrnTrlLog *etable.Table
	TstTrlLog *etable.Table
	TrnEpcLog *etable.Table
	TstEpcLog *etable.Table
	RunLog    *etable.Table

	TrainEnv env.FixedTable
	TestEnv  env.FixedTable

	Time *leabra.Time

	EDL    bool
	MemThr float64
}

func (s *Sim) PreTrainInit(BaseInfo string, dataName string) {
	ca3 := s.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	dg := s.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca3.Off = true
	dg.Off = true
	s.TrainEnv.Table = etable.NewIdxView(s.DataMap[dataName])
	s.TrainEnv.Init(0)
	for {
		if s.PreTrainInitTrail(BaseInfo) {
			break
		}
	}

	ca3.Off = false
	dg.Off = false
}

func (s *Sim) PreTrainInitTrail(BaseInfo string) bool {
	s.TrainEnv.Step()
	epc, _, chg := s.TrainEnv.Counter(env.Epoch)
	if chg {
		EpochStat(s.TrnTrlLog, s.TrnEpcLog, BaseInfo, int64(s.TrainEnv.Run.Cur), int64(s.TrainEnv.Epoch.Cur)-1, fmt.Sprint(s.Pa.ListSize), true)
		log.Println(EpochTableToStringLastLine(s.TrnEpcLog))
		if epc >= 5 {
			return true
		}
	}
	trailBaseInfo := &TrailBaseInfo{}
	trailBaseInfo.BaseStat(BaseInfo, int64(s.TrainEnv.Run.Cur), int64(s.TrainEnv.Epoch.Cur), int64(s.TrainEnv.Trial.Cur), s.TrainEnv.TrialName.Cur)

	s.applyInputs(true)
	s.AlphaCyc(trailBaseInfo, true)
	trailBaseInfo.SSEStat(s.Net)
	trailBaseInfo.SetToTrlLog(s.TrnTrlLog)
	log.Println(trailBaseInfo.ToString())
	return false
}

func (s *Sim) TrainRP(BaseInfo string, dataName string) {
	s.TrainEnv.Table = etable.NewIdxView(s.DataMap[dataName])
	s.TrainEnv.Init(0)
	s.TrainEnv.Trial.Cur = -1
	for {
		if s.RetrievalPracticeTrial(BaseInfo) {
			break
		}
	}
}

func (s *Sim) RetrievalPracticeTrial(BaseInfo string) bool {
	s.TrainEnv.Step()
	_, _, chg := s.TrainEnv.Counter(env.Epoch)
	if chg {
		EpochStat(s.TrnTrlLog, s.TrnEpcLog, BaseInfo, int64(s.TrainEnv.Run.Cur), int64(s.TrainEnv.Epoch.Cur)-1, fmt.Sprint(s.Pa.ListSize), true)
		log.Println(EpochTableToStringLastLine(s.TrnEpcLog))
		return true
	}
	trailBaseInfo := &TrailBaseInfo{}
	trailBaseInfo.BaseStat(BaseInfo, int64(s.TrainEnv.Run.Cur), int64(s.TrainEnv.Epoch.Cur), int64(s.TrainEnv.Trial.Cur), s.TrainEnv.TrialName.Cur)

	s.applyInputs(true)
	s.AlphaCycRP(trailBaseInfo, true)
	trailBaseInfo.SSEStat(s.Net)
	trailBaseInfo.SetToTrlLog(s.TrnTrlLog)
	log.Println(trailBaseInfo.ToString())
	return false
}

func (s *Sim) AlphaCycRP(trailBaseInfo *TrailBaseInfo, train bool) {
	if train {
		s.Net.WtFmDWt()
	}
	dg := s.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca1 := s.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := s.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	input := s.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecin := s.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := s.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(leabra.LeabraPrjn).AsLeabra()
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()
	_ = dg
	_ = ecin
	_ = input

	ecoutFmCa1 := ecout.RcvPrjns.SendName("CA1").(leabra.LeabraPrjn).AsLeabra()
	ca1FmECout := ca1.RcvPrjns.SendName("ECout").(leabra.LeabraPrjn).AsLeabra()
	ecoutFmCa1.Learn.Learn = false
	ca1FmECin.Learn.Learn = false
	ca1FmECout.Learn.Learn = false

	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0

	dgwtscale := ca3FmDg.WtScale.Rel

	if s.EDL == true {
		ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDel
	} else {
		ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDelTest
	}

	if train {
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		ecout.SetType(emer.Compare) // don't clamp
	}
	ecout.UpdateExtFlags()

	//s.Time.Reset()

	s.Net.AlphaCycInit(true)
	s.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < s.Time.CycPerQtr; cyc++ { //for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			s.Net.Cycle(s.Time)
			s.Time.CycleInc()
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1
			//ca3FmDg.WtScale.Rel = dgwtscale // RP: 4
			if !train { // zycyc: ???? RP IS testing
				ca3FmDg.WtScale.Rel = dgwtscale // 4
			} else {
				ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDelTest // RP: 1
			}
			s.Net.GScaleFmAvgAct() // update computed scaling factors
			s.Net.InitGInc()       // scaling params change, so need to recompute all netins                                         aaa

		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			s.Net.GScaleFmAvgAct() // update computed scaling factors
			s.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}
		s.Net.QuarterFinal(s.Time)
		if qtr+1 == 3 {
			trailBaseInfo.MemStat(s.Net, s.MemThr, train)
			trailBaseInfo.ErrorSignalStat(s.Net)
		}
		s.Time.QuarterInc()
	}

	ca3FmDg.WtScale.Rel = dgwtscale // restore
	ca1FmCa3.WtScale.Abs = 1

	if train {
		s.Net.DWt()
	}

}

func (s *Sim) Train(BaseInfo string, dataName string) {
	s.TrainEnv.Table = etable.NewIdxView(s.DataMap[dataName])
	s.TrainEnv.Init(0)
	s.TrainEnv.Trial.Cur = -1
	for {
		if s.TrainTrail(BaseInfo) {
			break
		}
	}
}

func (s *Sim) TrainTrail(BaseInfo string) bool {
	s.TrainEnv.Step()
	_, _, chg := s.TrainEnv.Counter(env.Epoch)
	if chg {
		EpochStat(s.TrnTrlLog, s.TrnEpcLog, BaseInfo, int64(s.TrainEnv.Run.Cur), int64(s.TrainEnv.Epoch.Cur)-1, fmt.Sprint(s.Pa.ListSize), true)
		log.Println(EpochTableToStringLastLine(s.TrnEpcLog))
		return true
	}
	trailBaseInfo := &TrailBaseInfo{}
	trailBaseInfo.BaseStat(BaseInfo, int64(s.TrainEnv.Run.Cur), int64(s.TrainEnv.Epoch.Cur), int64(s.TrainEnv.Trial.Cur), s.TrainEnv.TrialName.Cur)

	s.applyInputs(true)
	s.AlphaCyc(trailBaseInfo, true)
	trailBaseInfo.SSEStat(s.Net)
	trailBaseInfo.SetToTrlLog(s.TrnTrlLog)
	log.Println(trailBaseInfo.ToString())
	return false
}

func (s *Sim) AlphaCyc(trailBaseInfo *TrailBaseInfo, train bool) {
	dg := s.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca1 := s.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := s.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	input := s.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecin := s.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := s.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(leabra.LeabraPrjn).AsLeabra()
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()
	_ = dg
	_ = input
	_ = ecin

	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0

	dgwtscale := ca3FmDg.WtScale.Rel

	if train {
		ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDel
	} else {
		if s.EDL == true {
			ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDel
		} else {
			ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDelTest
		}
	}
	if train {
		ecout.SetType(emer.Target)
	} else {
		ecout.SetType(emer.Compare)
	}
	ecout.UpdateExtFlags()

	//s.Time.Reset()

	s.Net.AlphaCycInit(true)
	s.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < s.Time.CycPerQtr; cyc++ {
			s.Net.Cycle(s.Time)
			s.Time.CycleInc()
		}
		switch qtr + 1 {
		case 1:
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1
			if train {
				ca3FmDg.WtScale.Rel = dgwtscale
			} else {
				ca3FmDg.WtScale.Rel = dgwtscale - s.Hp.MossyDelTest
			}
			s.Net.GScaleFmAvgAct()
			s.Net.InitGInc()
		case 3:
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			s.Net.GScaleFmAvgAct()
			s.Net.InitGInc()
			if train {
				var TmpVals []float32
				ecin.UnitVals(&TmpVals, "Act")
				ecout.ApplyExt1D32(TmpVals)
			}
		}
		s.Net.QuarterFinal(s.Time)
		if qtr+1 == 3 {
			trailBaseInfo.MemStat(s.Net, s.MemThr, train)
			trailBaseInfo.ErrorSignalStat(s.Net)
		}
		s.Time.QuarterInc()

	}
	ca3FmDg.WtScale.Rel = dgwtscale
	ca1FmCa3.WtScale.Abs = 1

	if train {
		s.Net.DWt()
		s.Net.WtFmDWt()
	}
}

func (s *Sim) Test(BaseInfo string, dataName string) {
	s.TestEnv.Table = etable.NewIdxView(s.DataMap[dataName])
	s.TestEnv.Init(0)
	s.TestEnv.Trial.Cur = -1
	for {
		if s.TestTrial(BaseInfo) {
			break
		}
	}

}

func (s *Sim) TestTrial(BaseInfo string) bool {
	s.TestEnv.Step()

	_, _, chg := s.TestEnv.Counter(env.Epoch)
	if chg {
		EpochStat(s.TstTrlLog, s.TstEpcLog, BaseInfo, int64(s.TestEnv.Run.Cur), int64(s.TestEnv.Epoch.Cur)-1, fmt.Sprint(s.Pa.ListSize), false)
		log.Println(EpochTableToStringLastLine(s.TstEpcLog))
		return true
	}

	trailBaseInfo := &TrailBaseInfo{}
	trailBaseInfo.BaseStat(BaseInfo, int64(s.TestEnv.Run.Cur), int64(s.TestEnv.Epoch.Cur), int64(s.TestEnv.Trial.Cur), s.TestEnv.TrialName.Cur)

	s.applyInputs(false)
	s.AlphaCyc(trailBaseInfo, false)
	trailBaseInfo.SSEStat(s.Net)
	trailBaseInfo.SetToTrlLog(s.TstTrlLog)
	log.Println(trailBaseInfo.ToString())
	return false
}

func (s *Sim) applyInputs(train bool) {
	lays := []string{"Input", "ECout"}
	for _, lnm := range lays {
		ly := s.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		var pats etensor.Tensor
		if train {
			pats = s.TrainEnv.State(ly.Nm)
		} else {
			pats = s.TestEnv.State(ly.Nm)
		}
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}
