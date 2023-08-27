package model

import (
	"fmt"
	"github.com/emer/emergent/params"
	"testing"
)

func TestParamSet(t *testing.T) {
	hp := &HipParams{}
	hp.Defaults()

	var ParamSets = params.Sets{{Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
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
	pset, err := ParamSets.SetByNameTry("SmallHip")
	if err != nil {
		t.Fatal(err)
	}
	simp, ok := pset.Sheets["Hip"]
	if ok {
		simp.Apply(hp, true)
	}

	fmt.Println(hp)
}
