package train

import (
	"fmt"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/metric"
	"log"
	"os"
	"strconv"
	"testing"
)

func TestBaseTable(t *testing.T) {
	dt := &etable.Table{}
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(4))

	nt := 1000
	sch := etable.Schema{
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
		{"SignalError", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)

	fmt.Println(dt.ColNames, dt.Rows)

	dt.SetCellFloat("Run", 0, 1)
	dt.SetCellFloat("Epoch", 0, 1)
	dt.SetCellFloat("Trial", 0, 1)
	dt.SetCellString("TrialName", 0, "ab")

	fmt.Println("Run", dt.CellFloat("Run", 0))
	fmt.Println("Epoch", dt.CellFloat("Epoch", 0))
	fmt.Println("Trail", dt.CellFloat("Trial", 0))
	fmt.Println("TrailNm", dt.CellString("TrialName", 0))

	newCol := etensor.New(etensor.FLOAT64, []int{1000, 1}, nil, nil)
	err := dt.AddCol(newCol, "newCol")
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(dt.ColNames, dt.Rows)

	dt.AddRows(10)
	fmt.Println(dt.ColNames, dt.Rows)

}

func TestPValue(t *testing.T) {
	// 计算
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float64{2, 1, 4, 2, 4, 5, 6, 3}
	fmt.Println(metric.Correlation64(a, b))

}

func TestView(t *testing.T) {
	// 创建数据表
	dt := &etable.Table{}
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", "4")

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
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

	// 添加测试数据
	dt.SetNumRows(2)
	dt.SetCellFloat("Run", 0, 0)
	dt.SetCellFloat("Epoch", 0, 0)
	dt.SetCellString("TrialName", 0, "trial1")
	dt.SetCellFloat("SSE", 0, 0.1)
	dt.SetCellFloat("AvgSSE", 0, 0.2)
	dt.SetCellFloat("CosDiff", 0, 0.3)
	dt.SetCellFloat("Mem", 0, 0.4)
	dt.SetCellFloat("TrgOnWasOff", 0, 0.5)
	dt.SetCellFloat("TrgOffWasOn", 0, 0.6)
	dt.SetCellFloat("ErrorSignal", 0, 0.7)

	dt.SetCellFloat("Run", 1, 1)
	dt.SetCellFloat("Epoch", 1, 0)
	dt.SetCellString("TrialName", 1, "trial2")
	dt.SetCellFloat("SSE", 1, 0.2)
	dt.SetCellFloat("AvgSSE", 1, 0.3)
	dt.SetCellFloat("CosDiff", 1, 0.4)
	dt.SetCellFloat("Mem", 1, 0.5)
	dt.SetCellFloat("TrgOnWasOff", 1, 0.6)
	dt.SetCellFloat("TrgOffWasOn", 1, 0.7)
	dt.SetCellFloat("ErrorSignal", 1, 0.9)

	// 筛选符合条件的行
	idx := etable.NewIdxView(dt) // 创建视图

	// 筛选 Run 为 0，Epoch 为 0 的行
	idx.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Run", row) == 0 && et.CellFloat("Epoch", row) == 0
	})

	// 计算每一列的平均值
	mean := make([]float64, dt.NumCols())
	for i := 0; i < dt.NumCols(); i++ {
		mean[i] = agg.Mean(idx, dt.ColNames[i])[0]
	}

	// 输出结果
	fmt.Println("Mean values:")
	for i, val := range mean {
		fmt.Printf("%s: %f\n", dt.ColNames[i], val)
	}

	file, err := os.Create("output.csv")
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	defer file.Close()

	err = dt.WriteCSV(file, etable.Comma, true)
	if err != nil {
		t.Fatal(err)
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
