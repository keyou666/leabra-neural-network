package train

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"strconv"
)

func CreateTrainTrailLogTable() *etable.Table {
	dt := &etable.Table{}
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(4))

	sch := etable.Schema{
		{"BaseInfo", etensor.STRING, nil, nil},
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
		{"ErrorSignal", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)

	return dt
}

func CreateTrainEpochLogTable() *etable.Table {
	dt := &etable.Table{}
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(4))

	sch := etable.Schema{
		{"BaseInfo", etensor.STRING, nil, nil},
		{"NetSize", etensor.STRING, nil, nil},
		{"ListSize", etensor.STRING, nil, nil},
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
		{"AvgErrorSignal", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
	return dt
}

func CreateTestTrailLogTable() *etable.Table {
	dt := &etable.Table{}
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(4))

	sch := etable.Schema{
		{"BaseInfo", etensor.STRING, nil, nil},
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}

	dt.SetFromSchema(sch, 0)
	return dt
}

func CreateTestEpochLogTable() *etable.Table {
	dt := &etable.Table{}
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(4))

	sch := etable.Schema{
		{"BaseInfo", etensor.STRING, nil, nil},
		{"NetSize", etensor.STRING, nil, nil},
		{"ListSize", etensor.STRING, nil, nil},
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"ABMem", etensor.FLOAT64, nil, nil},
	}

	dt.SetFromSchema(sch, 0)
	return dt
}

func CreateRunLogTable() *etable.Table {
	dt := &etable.Table{}
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(4))

	sch := etable.Schema{
		//{"Params", etensor.STRING, nil, nil},
		{"NetSize", etensor.STRING, nil, nil},
		{"ListSize", etensor.STRING, nil, nil},
		{"Run", etensor.INT64, nil, nil},
		{"NEpochs", etensor.FLOAT64, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"ABMem", etensor.FLOAT64, nil, nil},
	}

	dt.SetFromSchema(sch, 0)
	return dt
}
