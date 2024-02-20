.DEFAULT_GOAL:=all
all : 

.PHONY: run-netload-S3-BG
run-netload-S3-BG : output/netload/mean-S3-BG.csv output/netload/std-S3-BG.csv output/netload/train-time-S3-BG output/netload/test-time-S3-BG output/netload/tree-S3-BG.svg

all : output/netload/mean-S3-BG.csv output/netload/std-S3-BG.csv output/netload/train-time-S3-BG output/netload/test-time-S3-BG output/netload/tree-S3-BG.svg

output/netload/mean-S3-BG.csv output/netload/std-S3-BG.csv output/netload/train-time-S3-BG output/netload/test-time-S3-BG output/netload/tree-S3-BG.svg &: settings/netload/S3-BG.toml netload_train/feature-S3-BG.csv netload_train/target-S3-BG.csv netload_test/feature-S3-BG.csv
	./run_model.py --config=settings/netload/S3-BG.toml --train-feature=netload_train/feature-S3-BG.csv --train-target=netload_train/target-S3-BG.csv --test-feature=netload_test/feature-S3-BG.csv --predict-mean=output/netload/mean-S3-BG.csv --predict-std=output/netload/std-S3-BG.csv --visualize-tree=output/netload/tree-S3-BG.svg --train-time=output/netload/train-time-S3-BG --test-time=output/netload/test-time-S3-BG 

.PHONY: show-netload-S3-BG
show-netload-S3-BG : output/netload/mean-S3-BG.csv output/netload/std-S3-BG.csv netload_test/target-S3-BG.csv
	./plot_prediction.py --mean=output/netload/mean-S3-BG.csv --std=output/netload/std-S3-BG.csv --target=netload_test/target-S3-BG.csv 

.PHONY: run-netload-S3-NL
run-netload-S3-NL : output/netload/mean-S3-NL.csv output/netload/std-S3-NL.csv output/netload/train-time-S3-NL output/netload/test-time-S3-NL output/netload/tree-S3-NL.svg

all : output/netload/mean-S3-NL.csv output/netload/std-S3-NL.csv output/netload/train-time-S3-NL output/netload/test-time-S3-NL output/netload/tree-S3-NL.svg

output/netload/mean-S3-NL.csv output/netload/std-S3-NL.csv output/netload/train-time-S3-NL output/netload/test-time-S3-NL output/netload/tree-S3-NL.svg &: settings/netload/S3-NL.toml netload_train/feature-S3-NL.csv netload_train/target-S3-NL.csv netload_test/feature-S3-NL.csv
	./run_model.py --config=settings/netload/S3-NL.toml --train-feature=netload_train/feature-S3-NL.csv --train-target=netload_train/target-S3-NL.csv --test-feature=netload_test/feature-S3-NL.csv --predict-mean=output/netload/mean-S3-NL.csv --predict-std=output/netload/std-S3-NL.csv --visualize-tree=output/netload/tree-S3-NL.svg --train-time=output/netload/train-time-S3-NL --test-time=output/netload/test-time-S3-NL 

.PHONY: show-netload-S3-NL
show-netload-S3-NL : output/netload/mean-S3-NL.csv output/netload/std-S3-NL.csv netload_test/target-S3-NL.csv
	./plot_prediction.py --mean=output/netload/mean-S3-NL.csv --std=output/netload/std-S3-NL.csv --target=netload_test/target-S3-NL.csv 

.PHONY: run-netload-S3-AT
run-netload-S3-AT : output/netload/mean-S3-AT.csv output/netload/std-S3-AT.csv output/netload/train-time-S3-AT output/netload/test-time-S3-AT output/netload/tree-S3-AT.svg

all : output/netload/mean-S3-AT.csv output/netload/std-S3-AT.csv output/netload/train-time-S3-AT output/netload/test-time-S3-AT output/netload/tree-S3-AT.svg

output/netload/mean-S3-AT.csv output/netload/std-S3-AT.csv output/netload/train-time-S3-AT output/netload/test-time-S3-AT output/netload/tree-S3-AT.svg &: settings/netload/S3-AT.toml netload_train/feature-S3-AT.csv netload_train/target-S3-AT.csv netload_test/feature-S3-AT.csv
	./run_model.py --config=settings/netload/S3-AT.toml --train-feature=netload_train/feature-S3-AT.csv --train-target=netload_train/target-S3-AT.csv --test-feature=netload_test/feature-S3-AT.csv --predict-mean=output/netload/mean-S3-AT.csv --predict-std=output/netload/std-S3-AT.csv --visualize-tree=output/netload/tree-S3-AT.svg --train-time=output/netload/train-time-S3-AT --test-time=output/netload/test-time-S3-AT 

.PHONY: show-netload-S3-AT
show-netload-S3-AT : output/netload/mean-S3-AT.csv output/netload/std-S3-AT.csv netload_test/target-S3-AT.csv
	./plot_prediction.py --mean=output/netload/mean-S3-AT.csv --std=output/netload/std-S3-AT.csv --target=netload_test/target-S3-AT.csv 

.PHONY: run-netload-S3-ES
run-netload-S3-ES : output/netload/mean-S3-ES.csv output/netload/std-S3-ES.csv output/netload/train-time-S3-ES output/netload/test-time-S3-ES output/netload/tree-S3-ES.svg

all : output/netload/mean-S3-ES.csv output/netload/std-S3-ES.csv output/netload/train-time-S3-ES output/netload/test-time-S3-ES output/netload/tree-S3-ES.svg

output/netload/mean-S3-ES.csv output/netload/std-S3-ES.csv output/netload/train-time-S3-ES output/netload/test-time-S3-ES output/netload/tree-S3-ES.svg &: settings/netload/S3-ES.toml netload_train/feature-S3-ES.csv netload_train/target-S3-ES.csv netload_test/feature-S3-ES.csv
	./run_model.py --config=settings/netload/S3-ES.toml --train-feature=netload_train/feature-S3-ES.csv --train-target=netload_train/target-S3-ES.csv --test-feature=netload_test/feature-S3-ES.csv --predict-mean=output/netload/mean-S3-ES.csv --predict-std=output/netload/std-S3-ES.csv --visualize-tree=output/netload/tree-S3-ES.svg --train-time=output/netload/train-time-S3-ES --test-time=output/netload/test-time-S3-ES 

.PHONY: show-netload-S3-ES
show-netload-S3-ES : output/netload/mean-S3-ES.csv output/netload/std-S3-ES.csv netload_test/target-S3-ES.csv
	./plot_prediction.py --mean=output/netload/mean-S3-ES.csv --std=output/netload/std-S3-ES.csv --target=netload_test/target-S3-ES.csv 

.PHONY: run-netload-S3-GR
run-netload-S3-GR : output/netload/mean-S3-GR.csv output/netload/std-S3-GR.csv output/netload/train-time-S3-GR output/netload/test-time-S3-GR output/netload/tree-S3-GR.svg

all : output/netload/mean-S3-GR.csv output/netload/std-S3-GR.csv output/netload/train-time-S3-GR output/netload/test-time-S3-GR output/netload/tree-S3-GR.svg

output/netload/mean-S3-GR.csv output/netload/std-S3-GR.csv output/netload/train-time-S3-GR output/netload/test-time-S3-GR output/netload/tree-S3-GR.svg &: settings/netload/S3-GR.toml netload_train/feature-S3-GR.csv netload_train/target-S3-GR.csv netload_test/feature-S3-GR.csv
	./run_model.py --config=settings/netload/S3-GR.toml --train-feature=netload_train/feature-S3-GR.csv --train-target=netload_train/target-S3-GR.csv --test-feature=netload_test/feature-S3-GR.csv --predict-mean=output/netload/mean-S3-GR.csv --predict-std=output/netload/std-S3-GR.csv --visualize-tree=output/netload/tree-S3-GR.svg --train-time=output/netload/train-time-S3-GR --test-time=output/netload/test-time-S3-GR 

.PHONY: show-netload-S3-GR
show-netload-S3-GR : output/netload/mean-S3-GR.csv output/netload/std-S3-GR.csv netload_test/target-S3-GR.csv
	./plot_prediction.py --mean=output/netload/mean-S3-GR.csv --std=output/netload/std-S3-GR.csv --target=netload_test/target-S3-GR.csv 

.PHONY: run-netload-S3-IT
run-netload-S3-IT : output/netload/mean-S3-IT.csv output/netload/std-S3-IT.csv output/netload/train-time-S3-IT output/netload/test-time-S3-IT output/netload/tree-S3-IT.svg

all : output/netload/mean-S3-IT.csv output/netload/std-S3-IT.csv output/netload/train-time-S3-IT output/netload/test-time-S3-IT output/netload/tree-S3-IT.svg

output/netload/mean-S3-IT.csv output/netload/std-S3-IT.csv output/netload/train-time-S3-IT output/netload/test-time-S3-IT output/netload/tree-S3-IT.svg &: settings/netload/S3-IT.toml netload_train/feature-S3-IT.csv netload_train/target-S3-IT.csv netload_test/feature-S3-IT.csv
	./run_model.py --config=settings/netload/S3-IT.toml --train-feature=netload_train/feature-S3-IT.csv --train-target=netload_train/target-S3-IT.csv --test-feature=netload_test/feature-S3-IT.csv --predict-mean=output/netload/mean-S3-IT.csv --predict-std=output/netload/std-S3-IT.csv --visualize-tree=output/netload/tree-S3-IT.svg --train-time=output/netload/train-time-S3-IT --test-time=output/netload/test-time-S3-IT 

.PHONY: show-netload-S3-IT
show-netload-S3-IT : output/netload/mean-S3-IT.csv output/netload/std-S3-IT.csv netload_test/target-S3-IT.csv
	./plot_prediction.py --mean=output/netload/mean-S3-IT.csv --std=output/netload/std-S3-IT.csv --target=netload_test/target-S3-IT.csv 

.PHONY: run-netload-S3-SI
run-netload-S3-SI : output/netload/mean-S3-SI.csv output/netload/std-S3-SI.csv output/netload/train-time-S3-SI output/netload/test-time-S3-SI output/netload/tree-S3-SI.svg

all : output/netload/mean-S3-SI.csv output/netload/std-S3-SI.csv output/netload/train-time-S3-SI output/netload/test-time-S3-SI output/netload/tree-S3-SI.svg

output/netload/mean-S3-SI.csv output/netload/std-S3-SI.csv output/netload/train-time-S3-SI output/netload/test-time-S3-SI output/netload/tree-S3-SI.svg &: settings/netload/S3-SI.toml netload_train/feature-S3-SI.csv netload_train/target-S3-SI.csv netload_test/feature-S3-SI.csv
	./run_model.py --config=settings/netload/S3-SI.toml --train-feature=netload_train/feature-S3-SI.csv --train-target=netload_train/target-S3-SI.csv --test-feature=netload_test/feature-S3-SI.csv --predict-mean=output/netload/mean-S3-SI.csv --predict-std=output/netload/std-S3-SI.csv --visualize-tree=output/netload/tree-S3-SI.svg --train-time=output/netload/train-time-S3-SI --test-time=output/netload/test-time-S3-SI 

.PHONY: show-netload-S3-SI
show-netload-S3-SI : output/netload/mean-S3-SI.csv output/netload/std-S3-SI.csv netload_test/target-S3-SI.csv
	./plot_prediction.py --mean=output/netload/mean-S3-SI.csv --std=output/netload/std-S3-SI.csv --target=netload_test/target-S3-SI.csv 

.PHONY: run-netload-S4-MIDATL
run-netload-S4-MIDATL : output/netload/mean-S4-MIDATL.csv output/netload/std-S4-MIDATL.csv output/netload/train-time-S4-MIDATL output/netload/test-time-S4-MIDATL output/netload/tree-S4-MIDATL.svg

all : output/netload/mean-S4-MIDATL.csv output/netload/std-S4-MIDATL.csv output/netload/train-time-S4-MIDATL output/netload/test-time-S4-MIDATL output/netload/tree-S4-MIDATL.svg

output/netload/mean-S4-MIDATL.csv output/netload/std-S4-MIDATL.csv output/netload/train-time-S4-MIDATL output/netload/test-time-S4-MIDATL output/netload/tree-S4-MIDATL.svg &: settings/netload/S4-MIDATL.toml netload_train/feature-S4-MIDATL.csv netload_train/target-S4-MIDATL.csv netload_test/feature-S4-MIDATL.csv
	./run_model.py --config=settings/netload/S4-MIDATL.toml --train-feature=netload_train/feature-S4-MIDATL.csv --train-target=netload_train/target-S4-MIDATL.csv --test-feature=netload_test/feature-S4-MIDATL.csv --predict-mean=output/netload/mean-S4-MIDATL.csv --predict-std=output/netload/std-S4-MIDATL.csv --visualize-tree=output/netload/tree-S4-MIDATL.svg --train-time=output/netload/train-time-S4-MIDATL --test-time=output/netload/test-time-S4-MIDATL 

.PHONY: show-netload-S4-MIDATL
show-netload-S4-MIDATL : output/netload/mean-S4-MIDATL.csv output/netload/std-S4-MIDATL.csv netload_test/target-S4-MIDATL.csv
	./plot_prediction.py --mean=output/netload/mean-S4-MIDATL.csv --std=output/netload/std-S4-MIDATL.csv --target=netload_test/target-S4-MIDATL.csv 

.PHONY: run-netload-S3-PT
run-netload-S3-PT : output/netload/mean-S3-PT.csv output/netload/std-S3-PT.csv output/netload/train-time-S3-PT output/netload/test-time-S3-PT output/netload/tree-S3-PT.svg

all : output/netload/mean-S3-PT.csv output/netload/std-S3-PT.csv output/netload/train-time-S3-PT output/netload/test-time-S3-PT output/netload/tree-S3-PT.svg

output/netload/mean-S3-PT.csv output/netload/std-S3-PT.csv output/netload/train-time-S3-PT output/netload/test-time-S3-PT output/netload/tree-S3-PT.svg &: settings/netload/S3-PT.toml netload_train/feature-S3-PT.csv netload_train/target-S3-PT.csv netload_test/feature-S3-PT.csv
	./run_model.py --config=settings/netload/S3-PT.toml --train-feature=netload_train/feature-S3-PT.csv --train-target=netload_train/target-S3-PT.csv --test-feature=netload_test/feature-S3-PT.csv --predict-mean=output/netload/mean-S3-PT.csv --predict-std=output/netload/std-S3-PT.csv --visualize-tree=output/netload/tree-S3-PT.svg --train-time=output/netload/train-time-S3-PT --test-time=output/netload/test-time-S3-PT 

.PHONY: show-netload-S3-PT
show-netload-S3-PT : output/netload/mean-S3-PT.csv output/netload/std-S3-PT.csv netload_test/target-S3-PT.csv
	./plot_prediction.py --mean=output/netload/mean-S3-PT.csv --std=output/netload/std-S3-PT.csv --target=netload_test/target-S3-PT.csv 

.PHONY: run-netload-S1
run-netload-S1 : output/netload/mean-S1.csv output/netload/std-S1.csv output/netload/train-time-S1 output/netload/test-time-S1 output/netload/tree-S1.svg

all : output/netload/mean-S1.csv output/netload/std-S1.csv output/netload/train-time-S1 output/netload/test-time-S1 output/netload/tree-S1.svg

output/netload/mean-S1.csv output/netload/std-S1.csv output/netload/train-time-S1 output/netload/test-time-S1 output/netload/tree-S1.svg &: settings/netload/S1.toml netload_train/feature-S1.csv netload_train/target-S1.csv netload_test/feature-S1.csv
	./run_model.py --config=settings/netload/S1.toml --train-feature=netload_train/feature-S1.csv --train-target=netload_train/target-S1.csv --test-feature=netload_test/feature-S1.csv --predict-mean=output/netload/mean-S1.csv --predict-std=output/netload/std-S1.csv --visualize-tree=output/netload/tree-S1.svg --train-time=output/netload/train-time-S1 --test-time=output/netload/test-time-S1 

.PHONY: show-netload-S1
show-netload-S1 : output/netload/mean-S1.csv output/netload/std-S1.csv netload_test/target-S1.csv
	./plot_prediction.py --mean=output/netload/mean-S1.csv --std=output/netload/std-S1.csv --target=netload_test/target-S1.csv 

.PHONY: run-netload-S3-CH
run-netload-S3-CH : output/netload/mean-S3-CH.csv output/netload/std-S3-CH.csv output/netload/train-time-S3-CH output/netload/test-time-S3-CH output/netload/tree-S3-CH.svg

all : output/netload/mean-S3-CH.csv output/netload/std-S3-CH.csv output/netload/train-time-S3-CH output/netload/test-time-S3-CH output/netload/tree-S3-CH.svg

output/netload/mean-S3-CH.csv output/netload/std-S3-CH.csv output/netload/train-time-S3-CH output/netload/test-time-S3-CH output/netload/tree-S3-CH.svg &: settings/netload/S3-CH.toml netload_train/feature-S3-CH.csv netload_train/target-S3-CH.csv netload_test/feature-S3-CH.csv
	./run_model.py --config=settings/netload/S3-CH.toml --train-feature=netload_train/feature-S3-CH.csv --train-target=netload_train/target-S3-CH.csv --test-feature=netload_test/feature-S3-CH.csv --predict-mean=output/netload/mean-S3-CH.csv --predict-std=output/netload/std-S3-CH.csv --visualize-tree=output/netload/tree-S3-CH.svg --train-time=output/netload/train-time-S3-CH --test-time=output/netload/test-time-S3-CH 

.PHONY: show-netload-S3-CH
show-netload-S3-CH : output/netload/mean-S3-CH.csv output/netload/std-S3-CH.csv netload_test/target-S3-CH.csv
	./plot_prediction.py --mean=output/netload/mean-S3-CH.csv --std=output/netload/std-S3-CH.csv --target=netload_test/target-S3-CH.csv 

.PHONY: run-netload-S3-SK
run-netload-S3-SK : output/netload/mean-S3-SK.csv output/netload/std-S3-SK.csv output/netload/train-time-S3-SK output/netload/test-time-S3-SK output/netload/tree-S3-SK.svg

all : output/netload/mean-S3-SK.csv output/netload/std-S3-SK.csv output/netload/train-time-S3-SK output/netload/test-time-S3-SK output/netload/tree-S3-SK.svg

output/netload/mean-S3-SK.csv output/netload/std-S3-SK.csv output/netload/train-time-S3-SK output/netload/test-time-S3-SK output/netload/tree-S3-SK.svg &: settings/netload/S3-SK.toml netload_train/feature-S3-SK.csv netload_train/target-S3-SK.csv netload_test/feature-S3-SK.csv
	./run_model.py --config=settings/netload/S3-SK.toml --train-feature=netload_train/feature-S3-SK.csv --train-target=netload_train/target-S3-SK.csv --test-feature=netload_test/feature-S3-SK.csv --predict-mean=output/netload/mean-S3-SK.csv --predict-std=output/netload/std-S3-SK.csv --visualize-tree=output/netload/tree-S3-SK.svg --train-time=output/netload/train-time-S3-SK --test-time=output/netload/test-time-S3-SK 

.PHONY: show-netload-S3-SK
show-netload-S3-SK : output/netload/mean-S3-SK.csv output/netload/std-S3-SK.csv netload_test/target-S3-SK.csv
	./plot_prediction.py --mean=output/netload/mean-S3-SK.csv --std=output/netload/std-S3-SK.csv --target=netload_test/target-S3-SK.csv 

.PHONY: run-netload-S3-DK
run-netload-S3-DK : output/netload/mean-S3-DK.csv output/netload/std-S3-DK.csv output/netload/train-time-S3-DK output/netload/test-time-S3-DK output/netload/tree-S3-DK.svg

all : output/netload/mean-S3-DK.csv output/netload/std-S3-DK.csv output/netload/train-time-S3-DK output/netload/test-time-S3-DK output/netload/tree-S3-DK.svg

output/netload/mean-S3-DK.csv output/netload/std-S3-DK.csv output/netload/train-time-S3-DK output/netload/test-time-S3-DK output/netload/tree-S3-DK.svg &: settings/netload/S3-DK.toml netload_train/feature-S3-DK.csv netload_train/target-S3-DK.csv netload_test/feature-S3-DK.csv
	./run_model.py --config=settings/netload/S3-DK.toml --train-feature=netload_train/feature-S3-DK.csv --train-target=netload_train/target-S3-DK.csv --test-feature=netload_test/feature-S3-DK.csv --predict-mean=output/netload/mean-S3-DK.csv --predict-std=output/netload/std-S3-DK.csv --visualize-tree=output/netload/tree-S3-DK.svg --train-time=output/netload/train-time-S3-DK --test-time=output/netload/test-time-S3-DK 

.PHONY: show-netload-S3-DK
show-netload-S3-DK : output/netload/mean-S3-DK.csv output/netload/std-S3-DK.csv netload_test/target-S3-DK.csv
	./plot_prediction.py --mean=output/netload/mean-S3-DK.csv --std=output/netload/std-S3-DK.csv --target=netload_test/target-S3-DK.csv 

.PHONY: run-netload-S3-FR
run-netload-S3-FR : output/netload/mean-S3-FR.csv output/netload/std-S3-FR.csv output/netload/train-time-S3-FR output/netload/test-time-S3-FR output/netload/tree-S3-FR.svg

all : output/netload/mean-S3-FR.csv output/netload/std-S3-FR.csv output/netload/train-time-S3-FR output/netload/test-time-S3-FR output/netload/tree-S3-FR.svg

output/netload/mean-S3-FR.csv output/netload/std-S3-FR.csv output/netload/train-time-S3-FR output/netload/test-time-S3-FR output/netload/tree-S3-FR.svg &: settings/netload/S3-FR.toml netload_train/feature-S3-FR.csv netload_train/target-S3-FR.csv netload_test/feature-S3-FR.csv
	./run_model.py --config=settings/netload/S3-FR.toml --train-feature=netload_train/feature-S3-FR.csv --train-target=netload_train/target-S3-FR.csv --test-feature=netload_test/feature-S3-FR.csv --predict-mean=output/netload/mean-S3-FR.csv --predict-std=output/netload/std-S3-FR.csv --visualize-tree=output/netload/tree-S3-FR.svg --train-time=output/netload/train-time-S3-FR --test-time=output/netload/test-time-S3-FR 

.PHONY: show-netload-S3-FR
show-netload-S3-FR : output/netload/mean-S3-FR.csv output/netload/std-S3-FR.csv netload_test/target-S3-FR.csv
	./plot_prediction.py --mean=output/netload/mean-S3-FR.csv --std=output/netload/std-S3-FR.csv --target=netload_test/target-S3-FR.csv 

.PHONY: run-netload-S3-BE
run-netload-S3-BE : output/netload/mean-S3-BE.csv output/netload/std-S3-BE.csv output/netload/train-time-S3-BE output/netload/test-time-S3-BE output/netload/tree-S3-BE.svg

all : output/netload/mean-S3-BE.csv output/netload/std-S3-BE.csv output/netload/train-time-S3-BE output/netload/test-time-S3-BE output/netload/tree-S3-BE.svg

output/netload/mean-S3-BE.csv output/netload/std-S3-BE.csv output/netload/train-time-S3-BE output/netload/test-time-S3-BE output/netload/tree-S3-BE.svg &: settings/netload/S3-BE.toml netload_train/feature-S3-BE.csv netload_train/target-S3-BE.csv netload_test/feature-S3-BE.csv
	./run_model.py --config=settings/netload/S3-BE.toml --train-feature=netload_train/feature-S3-BE.csv --train-target=netload_train/target-S3-BE.csv --test-feature=netload_test/feature-S3-BE.csv --predict-mean=output/netload/mean-S3-BE.csv --predict-std=output/netload/std-S3-BE.csv --visualize-tree=output/netload/tree-S3-BE.svg --train-time=output/netload/train-time-S3-BE --test-time=output/netload/test-time-S3-BE 

.PHONY: show-netload-S3-BE
show-netload-S3-BE : output/netload/mean-S3-BE.csv output/netload/std-S3-BE.csv netload_test/target-S3-BE.csv
	./plot_prediction.py --mean=output/netload/mean-S3-BE.csv --std=output/netload/std-S3-BE.csv --target=netload_test/target-S3-BE.csv 

.PHONY: run-netload-S4-WEST
run-netload-S4-WEST : output/netload/mean-S4-WEST.csv output/netload/std-S4-WEST.csv output/netload/train-time-S4-WEST output/netload/test-time-S4-WEST output/netload/tree-S4-WEST.svg

all : output/netload/mean-S4-WEST.csv output/netload/std-S4-WEST.csv output/netload/train-time-S4-WEST output/netload/test-time-S4-WEST output/netload/tree-S4-WEST.svg

output/netload/mean-S4-WEST.csv output/netload/std-S4-WEST.csv output/netload/train-time-S4-WEST output/netload/test-time-S4-WEST output/netload/tree-S4-WEST.svg &: settings/netload/S4-WEST.toml netload_train/feature-S4-WEST.csv netload_train/target-S4-WEST.csv netload_test/feature-S4-WEST.csv
	./run_model.py --config=settings/netload/S4-WEST.toml --train-feature=netload_train/feature-S4-WEST.csv --train-target=netload_train/target-S4-WEST.csv --test-feature=netload_test/feature-S4-WEST.csv --predict-mean=output/netload/mean-S4-WEST.csv --predict-std=output/netload/std-S4-WEST.csv --visualize-tree=output/netload/tree-S4-WEST.svg --train-time=output/netload/train-time-S4-WEST --test-time=output/netload/test-time-S4-WEST 

.PHONY: show-netload-S4-WEST
show-netload-S4-WEST : output/netload/mean-S4-WEST.csv output/netload/std-S4-WEST.csv netload_test/target-S4-WEST.csv
	./plot_prediction.py --mean=output/netload/mean-S4-WEST.csv --std=output/netload/std-S4-WEST.csv --target=netload_test/target-S4-WEST.csv 

.PHONY: run-netload-S2
run-netload-S2 : output/netload/mean-S2.csv output/netload/std-S2.csv output/netload/train-time-S2 output/netload/test-time-S2 output/netload/tree-S2.svg

all : output/netload/mean-S2.csv output/netload/std-S2.csv output/netload/train-time-S2 output/netload/test-time-S2 output/netload/tree-S2.svg

output/netload/mean-S2.csv output/netload/std-S2.csv output/netload/train-time-S2 output/netload/test-time-S2 output/netload/tree-S2.svg &: settings/netload/S2.toml netload_train/feature-S2.csv netload_train/target-S2.csv netload_test/feature-S2.csv
	./run_model.py --config=settings/netload/S2.toml --train-feature=netload_train/feature-S2.csv --train-target=netload_train/target-S2.csv --test-feature=netload_test/feature-S2.csv --predict-mean=output/netload/mean-S2.csv --predict-std=output/netload/std-S2.csv --visualize-tree=output/netload/tree-S2.svg --train-time=output/netload/train-time-S2 --test-time=output/netload/test-time-S2 

.PHONY: show-netload-S2
show-netload-S2 : output/netload/mean-S2.csv output/netload/std-S2.csv netload_test/target-S2.csv
	./plot_prediction.py --mean=output/netload/mean-S2.csv --std=output/netload/std-S2.csv --target=netload_test/target-S2.csv 

.PHONY: run-netload-S3-CZ
run-netload-S3-CZ : output/netload/mean-S3-CZ.csv output/netload/std-S3-CZ.csv output/netload/train-time-S3-CZ output/netload/test-time-S3-CZ output/netload/tree-S3-CZ.svg

all : output/netload/mean-S3-CZ.csv output/netload/std-S3-CZ.csv output/netload/train-time-S3-CZ output/netload/test-time-S3-CZ output/netload/tree-S3-CZ.svg

output/netload/mean-S3-CZ.csv output/netload/std-S3-CZ.csv output/netload/train-time-S3-CZ output/netload/test-time-S3-CZ output/netload/tree-S3-CZ.svg &: settings/netload/S3-CZ.toml netload_train/feature-S3-CZ.csv netload_train/target-S3-CZ.csv netload_test/feature-S3-CZ.csv
	./run_model.py --config=settings/netload/S3-CZ.toml --train-feature=netload_train/feature-S3-CZ.csv --train-target=netload_train/target-S3-CZ.csv --test-feature=netload_test/feature-S3-CZ.csv --predict-mean=output/netload/mean-S3-CZ.csv --predict-std=output/netload/std-S3-CZ.csv --visualize-tree=output/netload/tree-S3-CZ.svg --train-time=output/netload/train-time-S3-CZ --test-time=output/netload/test-time-S3-CZ 

.PHONY: show-netload-S3-CZ
show-netload-S3-CZ : output/netload/mean-S3-CZ.csv output/netload/std-S3-CZ.csv netload_test/target-S3-CZ.csv
	./plot_prediction.py --mean=output/netload/mean-S3-CZ.csv --std=output/netload/std-S3-CZ.csv --target=netload_test/target-S3-CZ.csv 

.PHONY: run-netload-S4-SOUTH
run-netload-S4-SOUTH : output/netload/mean-S4-SOUTH.csv output/netload/std-S4-SOUTH.csv output/netload/train-time-S4-SOUTH output/netload/test-time-S4-SOUTH output/netload/tree-S4-SOUTH.svg

all : output/netload/mean-S4-SOUTH.csv output/netload/std-S4-SOUTH.csv output/netload/train-time-S4-SOUTH output/netload/test-time-S4-SOUTH output/netload/tree-S4-SOUTH.svg

output/netload/mean-S4-SOUTH.csv output/netload/std-S4-SOUTH.csv output/netload/train-time-S4-SOUTH output/netload/test-time-S4-SOUTH output/netload/tree-S4-SOUTH.svg &: settings/netload/S4-SOUTH.toml netload_train/feature-S4-SOUTH.csv netload_train/target-S4-SOUTH.csv netload_test/feature-S4-SOUTH.csv
	./run_model.py --config=settings/netload/S4-SOUTH.toml --train-feature=netload_train/feature-S4-SOUTH.csv --train-target=netload_train/target-S4-SOUTH.csv --test-feature=netload_test/feature-S4-SOUTH.csv --predict-mean=output/netload/mean-S4-SOUTH.csv --predict-std=output/netload/std-S4-SOUTH.csv --visualize-tree=output/netload/tree-S4-SOUTH.svg --train-time=output/netload/train-time-S4-SOUTH --test-time=output/netload/test-time-S4-SOUTH 

.PHONY: show-netload-S4-SOUTH
show-netload-S4-SOUTH : output/netload/mean-S4-SOUTH.csv output/netload/std-S4-SOUTH.csv netload_test/target-S4-SOUTH.csv
	./plot_prediction.py --mean=output/netload/mean-S4-SOUTH.csv --std=output/netload/std-S4-SOUTH.csv --target=netload_test/target-S4-SOUTH.csv 

.PHONY: run-potential-S3-BG
run-potential-S3-BG : output/potential/mean-S3-BG.csv output/potential/std-S3-BG.csv output/potential/train-time-S3-BG output/potential/test-time-S3-BG output/potential/tree-S3-BG.svg

all : output/potential/mean-S3-BG.csv output/potential/std-S3-BG.csv output/potential/train-time-S3-BG output/potential/test-time-S3-BG output/potential/tree-S3-BG.svg

output/potential/mean-S3-BG.csv output/potential/std-S3-BG.csv output/potential/train-time-S3-BG output/potential/test-time-S3-BG output/potential/tree-S3-BG.svg &: settings/potential/S3-BG.toml potential_train/feature-S3-BG.csv potential_train/target-S3-BG.csv potential_test/feature-S3-BG.csv
	./run_model.py --config=settings/potential/S3-BG.toml --train-feature=potential_train/feature-S3-BG.csv --train-target=potential_train/target-S3-BG.csv --test-feature=potential_test/feature-S3-BG.csv --predict-mean=output/potential/mean-S3-BG.csv --predict-std=output/potential/std-S3-BG.csv --visualize-tree=output/potential/tree-S3-BG.svg --train-time=output/potential/train-time-S3-BG --test-time=output/potential/test-time-S3-BG 

.PHONY: show-potential-S3-BG
show-potential-S3-BG : output/potential/mean-S3-BG.csv output/potential/std-S3-BG.csv potential_test/target-S3-BG.csv
	./plot_prediction.py --mean=output/potential/mean-S3-BG.csv --std=output/potential/std-S3-BG.csv --target=potential_test/target-S3-BG.csv 

.PHONY: run-potential-S3-NL
run-potential-S3-NL : output/potential/mean-S3-NL.csv output/potential/std-S3-NL.csv output/potential/train-time-S3-NL output/potential/test-time-S3-NL output/potential/tree-S3-NL.svg

all : output/potential/mean-S3-NL.csv output/potential/std-S3-NL.csv output/potential/train-time-S3-NL output/potential/test-time-S3-NL output/potential/tree-S3-NL.svg

output/potential/mean-S3-NL.csv output/potential/std-S3-NL.csv output/potential/train-time-S3-NL output/potential/test-time-S3-NL output/potential/tree-S3-NL.svg &: settings/potential/S3-NL.toml potential_train/feature-S3-NL.csv potential_train/target-S3-NL.csv potential_test/feature-S3-NL.csv
	./run_model.py --config=settings/potential/S3-NL.toml --train-feature=potential_train/feature-S3-NL.csv --train-target=potential_train/target-S3-NL.csv --test-feature=potential_test/feature-S3-NL.csv --predict-mean=output/potential/mean-S3-NL.csv --predict-std=output/potential/std-S3-NL.csv --visualize-tree=output/potential/tree-S3-NL.svg --train-time=output/potential/train-time-S3-NL --test-time=output/potential/test-time-S3-NL 

.PHONY: show-potential-S3-NL
show-potential-S3-NL : output/potential/mean-S3-NL.csv output/potential/std-S3-NL.csv potential_test/target-S3-NL.csv
	./plot_prediction.py --mean=output/potential/mean-S3-NL.csv --std=output/potential/std-S3-NL.csv --target=potential_test/target-S3-NL.csv 

.PHONY: run-potential-S3-AT
run-potential-S3-AT : output/potential/mean-S3-AT.csv output/potential/std-S3-AT.csv output/potential/train-time-S3-AT output/potential/test-time-S3-AT output/potential/tree-S3-AT.svg

all : output/potential/mean-S3-AT.csv output/potential/std-S3-AT.csv output/potential/train-time-S3-AT output/potential/test-time-S3-AT output/potential/tree-S3-AT.svg

output/potential/mean-S3-AT.csv output/potential/std-S3-AT.csv output/potential/train-time-S3-AT output/potential/test-time-S3-AT output/potential/tree-S3-AT.svg &: settings/potential/S3-AT.toml potential_train/feature-S3-AT.csv potential_train/target-S3-AT.csv potential_test/feature-S3-AT.csv
	./run_model.py --config=settings/potential/S3-AT.toml --train-feature=potential_train/feature-S3-AT.csv --train-target=potential_train/target-S3-AT.csv --test-feature=potential_test/feature-S3-AT.csv --predict-mean=output/potential/mean-S3-AT.csv --predict-std=output/potential/std-S3-AT.csv --visualize-tree=output/potential/tree-S3-AT.svg --train-time=output/potential/train-time-S3-AT --test-time=output/potential/test-time-S3-AT 

.PHONY: show-potential-S3-AT
show-potential-S3-AT : output/potential/mean-S3-AT.csv output/potential/std-S3-AT.csv potential_test/target-S3-AT.csv
	./plot_prediction.py --mean=output/potential/mean-S3-AT.csv --std=output/potential/std-S3-AT.csv --target=potential_test/target-S3-AT.csv 

.PHONY: run-potential-S3-ES
run-potential-S3-ES : output/potential/mean-S3-ES.csv output/potential/std-S3-ES.csv output/potential/train-time-S3-ES output/potential/test-time-S3-ES output/potential/tree-S3-ES.svg

all : output/potential/mean-S3-ES.csv output/potential/std-S3-ES.csv output/potential/train-time-S3-ES output/potential/test-time-S3-ES output/potential/tree-S3-ES.svg

output/potential/mean-S3-ES.csv output/potential/std-S3-ES.csv output/potential/train-time-S3-ES output/potential/test-time-S3-ES output/potential/tree-S3-ES.svg &: settings/potential/S3-ES.toml potential_train/feature-S3-ES.csv potential_train/target-S3-ES.csv potential_test/feature-S3-ES.csv
	./run_model.py --config=settings/potential/S3-ES.toml --train-feature=potential_train/feature-S3-ES.csv --train-target=potential_train/target-S3-ES.csv --test-feature=potential_test/feature-S3-ES.csv --predict-mean=output/potential/mean-S3-ES.csv --predict-std=output/potential/std-S3-ES.csv --visualize-tree=output/potential/tree-S3-ES.svg --train-time=output/potential/train-time-S3-ES --test-time=output/potential/test-time-S3-ES 

.PHONY: show-potential-S3-ES
show-potential-S3-ES : output/potential/mean-S3-ES.csv output/potential/std-S3-ES.csv potential_test/target-S3-ES.csv
	./plot_prediction.py --mean=output/potential/mean-S3-ES.csv --std=output/potential/std-S3-ES.csv --target=potential_test/target-S3-ES.csv 

.PHONY: run-potential-S3-GR
run-potential-S3-GR : output/potential/mean-S3-GR.csv output/potential/std-S3-GR.csv output/potential/train-time-S3-GR output/potential/test-time-S3-GR output/potential/tree-S3-GR.svg

all : output/potential/mean-S3-GR.csv output/potential/std-S3-GR.csv output/potential/train-time-S3-GR output/potential/test-time-S3-GR output/potential/tree-S3-GR.svg

output/potential/mean-S3-GR.csv output/potential/std-S3-GR.csv output/potential/train-time-S3-GR output/potential/test-time-S3-GR output/potential/tree-S3-GR.svg &: settings/potential/S3-GR.toml potential_train/feature-S3-GR.csv potential_train/target-S3-GR.csv potential_test/feature-S3-GR.csv
	./run_model.py --config=settings/potential/S3-GR.toml --train-feature=potential_train/feature-S3-GR.csv --train-target=potential_train/target-S3-GR.csv --test-feature=potential_test/feature-S3-GR.csv --predict-mean=output/potential/mean-S3-GR.csv --predict-std=output/potential/std-S3-GR.csv --visualize-tree=output/potential/tree-S3-GR.svg --train-time=output/potential/train-time-S3-GR --test-time=output/potential/test-time-S3-GR 

.PHONY: show-potential-S3-GR
show-potential-S3-GR : output/potential/mean-S3-GR.csv output/potential/std-S3-GR.csv potential_test/target-S3-GR.csv
	./plot_prediction.py --mean=output/potential/mean-S3-GR.csv --std=output/potential/std-S3-GR.csv --target=potential_test/target-S3-GR.csv 

.PHONY: run-potential-S3-IT
run-potential-S3-IT : output/potential/mean-S3-IT.csv output/potential/std-S3-IT.csv output/potential/train-time-S3-IT output/potential/test-time-S3-IT output/potential/tree-S3-IT.svg

all : output/potential/mean-S3-IT.csv output/potential/std-S3-IT.csv output/potential/train-time-S3-IT output/potential/test-time-S3-IT output/potential/tree-S3-IT.svg

output/potential/mean-S3-IT.csv output/potential/std-S3-IT.csv output/potential/train-time-S3-IT output/potential/test-time-S3-IT output/potential/tree-S3-IT.svg &: settings/potential/S3-IT.toml potential_train/feature-S3-IT.csv potential_train/target-S3-IT.csv potential_test/feature-S3-IT.csv
	./run_model.py --config=settings/potential/S3-IT.toml --train-feature=potential_train/feature-S3-IT.csv --train-target=potential_train/target-S3-IT.csv --test-feature=potential_test/feature-S3-IT.csv --predict-mean=output/potential/mean-S3-IT.csv --predict-std=output/potential/std-S3-IT.csv --visualize-tree=output/potential/tree-S3-IT.svg --train-time=output/potential/train-time-S3-IT --test-time=output/potential/test-time-S3-IT 

.PHONY: show-potential-S3-IT
show-potential-S3-IT : output/potential/mean-S3-IT.csv output/potential/std-S3-IT.csv potential_test/target-S3-IT.csv
	./plot_prediction.py --mean=output/potential/mean-S3-IT.csv --std=output/potential/std-S3-IT.csv --target=potential_test/target-S3-IT.csv 

.PHONY: run-potential-S3-SI
run-potential-S3-SI : output/potential/mean-S3-SI.csv output/potential/std-S3-SI.csv output/potential/train-time-S3-SI output/potential/test-time-S3-SI output/potential/tree-S3-SI.svg

all : output/potential/mean-S3-SI.csv output/potential/std-S3-SI.csv output/potential/train-time-S3-SI output/potential/test-time-S3-SI output/potential/tree-S3-SI.svg

output/potential/mean-S3-SI.csv output/potential/std-S3-SI.csv output/potential/train-time-S3-SI output/potential/test-time-S3-SI output/potential/tree-S3-SI.svg &: settings/potential/S3-SI.toml potential_train/feature-S3-SI.csv potential_train/target-S3-SI.csv potential_test/feature-S3-SI.csv
	./run_model.py --config=settings/potential/S3-SI.toml --train-feature=potential_train/feature-S3-SI.csv --train-target=potential_train/target-S3-SI.csv --test-feature=potential_test/feature-S3-SI.csv --predict-mean=output/potential/mean-S3-SI.csv --predict-std=output/potential/std-S3-SI.csv --visualize-tree=output/potential/tree-S3-SI.svg --train-time=output/potential/train-time-S3-SI --test-time=output/potential/test-time-S3-SI 

.PHONY: show-potential-S3-SI
show-potential-S3-SI : output/potential/mean-S3-SI.csv output/potential/std-S3-SI.csv potential_test/target-S3-SI.csv
	./plot_prediction.py --mean=output/potential/mean-S3-SI.csv --std=output/potential/std-S3-SI.csv --target=potential_test/target-S3-SI.csv 

.PHONY: run-potential-S4-MIDATL
run-potential-S4-MIDATL : output/potential/mean-S4-MIDATL.csv output/potential/std-S4-MIDATL.csv output/potential/train-time-S4-MIDATL output/potential/test-time-S4-MIDATL output/potential/tree-S4-MIDATL.svg

all : output/potential/mean-S4-MIDATL.csv output/potential/std-S4-MIDATL.csv output/potential/train-time-S4-MIDATL output/potential/test-time-S4-MIDATL output/potential/tree-S4-MIDATL.svg

output/potential/mean-S4-MIDATL.csv output/potential/std-S4-MIDATL.csv output/potential/train-time-S4-MIDATL output/potential/test-time-S4-MIDATL output/potential/tree-S4-MIDATL.svg &: settings/potential/S4-MIDATL.toml potential_train/feature-S4-MIDATL.csv potential_train/target-S4-MIDATL.csv potential_test/feature-S4-MIDATL.csv
	./run_model.py --config=settings/potential/S4-MIDATL.toml --train-feature=potential_train/feature-S4-MIDATL.csv --train-target=potential_train/target-S4-MIDATL.csv --test-feature=potential_test/feature-S4-MIDATL.csv --predict-mean=output/potential/mean-S4-MIDATL.csv --predict-std=output/potential/std-S4-MIDATL.csv --visualize-tree=output/potential/tree-S4-MIDATL.svg --train-time=output/potential/train-time-S4-MIDATL --test-time=output/potential/test-time-S4-MIDATL 

.PHONY: show-potential-S4-MIDATL
show-potential-S4-MIDATL : output/potential/mean-S4-MIDATL.csv output/potential/std-S4-MIDATL.csv potential_test/target-S4-MIDATL.csv
	./plot_prediction.py --mean=output/potential/mean-S4-MIDATL.csv --std=output/potential/std-S4-MIDATL.csv --target=potential_test/target-S4-MIDATL.csv 

.PHONY: run-potential-S3-PT
run-potential-S3-PT : output/potential/mean-S3-PT.csv output/potential/std-S3-PT.csv output/potential/train-time-S3-PT output/potential/test-time-S3-PT output/potential/tree-S3-PT.svg

all : output/potential/mean-S3-PT.csv output/potential/std-S3-PT.csv output/potential/train-time-S3-PT output/potential/test-time-S3-PT output/potential/tree-S3-PT.svg

output/potential/mean-S3-PT.csv output/potential/std-S3-PT.csv output/potential/train-time-S3-PT output/potential/test-time-S3-PT output/potential/tree-S3-PT.svg &: settings/potential/S3-PT.toml potential_train/feature-S3-PT.csv potential_train/target-S3-PT.csv potential_test/feature-S3-PT.csv
	./run_model.py --config=settings/potential/S3-PT.toml --train-feature=potential_train/feature-S3-PT.csv --train-target=potential_train/target-S3-PT.csv --test-feature=potential_test/feature-S3-PT.csv --predict-mean=output/potential/mean-S3-PT.csv --predict-std=output/potential/std-S3-PT.csv --visualize-tree=output/potential/tree-S3-PT.svg --train-time=output/potential/train-time-S3-PT --test-time=output/potential/test-time-S3-PT 

.PHONY: show-potential-S3-PT
show-potential-S3-PT : output/potential/mean-S3-PT.csv output/potential/std-S3-PT.csv potential_test/target-S3-PT.csv
	./plot_prediction.py --mean=output/potential/mean-S3-PT.csv --std=output/potential/std-S3-PT.csv --target=potential_test/target-S3-PT.csv 

.PHONY: run-potential-S1
run-potential-S1 : output/potential/mean-S1.csv output/potential/std-S1.csv output/potential/train-time-S1 output/potential/test-time-S1 output/potential/tree-S1.svg

all : output/potential/mean-S1.csv output/potential/std-S1.csv output/potential/train-time-S1 output/potential/test-time-S1 output/potential/tree-S1.svg

output/potential/mean-S1.csv output/potential/std-S1.csv output/potential/train-time-S1 output/potential/test-time-S1 output/potential/tree-S1.svg &: settings/potential/S1.toml potential_train/feature-S1.csv potential_train/target-S1.csv potential_test/feature-S1.csv
	./run_model.py --config=settings/potential/S1.toml --train-feature=potential_train/feature-S1.csv --train-target=potential_train/target-S1.csv --test-feature=potential_test/feature-S1.csv --predict-mean=output/potential/mean-S1.csv --predict-std=output/potential/std-S1.csv --visualize-tree=output/potential/tree-S1.svg --train-time=output/potential/train-time-S1 --test-time=output/potential/test-time-S1 

.PHONY: show-potential-S1
show-potential-S1 : output/potential/mean-S1.csv output/potential/std-S1.csv potential_test/target-S1.csv
	./plot_prediction.py --mean=output/potential/mean-S1.csv --std=output/potential/std-S1.csv --target=potential_test/target-S1.csv 

.PHONY: run-potential-S3-CH
run-potential-S3-CH : output/potential/mean-S3-CH.csv output/potential/std-S3-CH.csv output/potential/train-time-S3-CH output/potential/test-time-S3-CH output/potential/tree-S3-CH.svg

all : output/potential/mean-S3-CH.csv output/potential/std-S3-CH.csv output/potential/train-time-S3-CH output/potential/test-time-S3-CH output/potential/tree-S3-CH.svg

output/potential/mean-S3-CH.csv output/potential/std-S3-CH.csv output/potential/train-time-S3-CH output/potential/test-time-S3-CH output/potential/tree-S3-CH.svg &: settings/potential/S3-CH.toml potential_train/feature-S3-CH.csv potential_train/target-S3-CH.csv potential_test/feature-S3-CH.csv
	./run_model.py --config=settings/potential/S3-CH.toml --train-feature=potential_train/feature-S3-CH.csv --train-target=potential_train/target-S3-CH.csv --test-feature=potential_test/feature-S3-CH.csv --predict-mean=output/potential/mean-S3-CH.csv --predict-std=output/potential/std-S3-CH.csv --visualize-tree=output/potential/tree-S3-CH.svg --train-time=output/potential/train-time-S3-CH --test-time=output/potential/test-time-S3-CH 

.PHONY: show-potential-S3-CH
show-potential-S3-CH : output/potential/mean-S3-CH.csv output/potential/std-S3-CH.csv potential_test/target-S3-CH.csv
	./plot_prediction.py --mean=output/potential/mean-S3-CH.csv --std=output/potential/std-S3-CH.csv --target=potential_test/target-S3-CH.csv 

.PHONY: run-potential-S3-SK
run-potential-S3-SK : output/potential/mean-S3-SK.csv output/potential/std-S3-SK.csv output/potential/train-time-S3-SK output/potential/test-time-S3-SK output/potential/tree-S3-SK.svg

all : output/potential/mean-S3-SK.csv output/potential/std-S3-SK.csv output/potential/train-time-S3-SK output/potential/test-time-S3-SK output/potential/tree-S3-SK.svg

output/potential/mean-S3-SK.csv output/potential/std-S3-SK.csv output/potential/train-time-S3-SK output/potential/test-time-S3-SK output/potential/tree-S3-SK.svg &: settings/potential/S3-SK.toml potential_train/feature-S3-SK.csv potential_train/target-S3-SK.csv potential_test/feature-S3-SK.csv
	./run_model.py --config=settings/potential/S3-SK.toml --train-feature=potential_train/feature-S3-SK.csv --train-target=potential_train/target-S3-SK.csv --test-feature=potential_test/feature-S3-SK.csv --predict-mean=output/potential/mean-S3-SK.csv --predict-std=output/potential/std-S3-SK.csv --visualize-tree=output/potential/tree-S3-SK.svg --train-time=output/potential/train-time-S3-SK --test-time=output/potential/test-time-S3-SK 

.PHONY: show-potential-S3-SK
show-potential-S3-SK : output/potential/mean-S3-SK.csv output/potential/std-S3-SK.csv potential_test/target-S3-SK.csv
	./plot_prediction.py --mean=output/potential/mean-S3-SK.csv --std=output/potential/std-S3-SK.csv --target=potential_test/target-S3-SK.csv 

.PHONY: run-potential-S3-DK
run-potential-S3-DK : output/potential/mean-S3-DK.csv output/potential/std-S3-DK.csv output/potential/train-time-S3-DK output/potential/test-time-S3-DK output/potential/tree-S3-DK.svg

all : output/potential/mean-S3-DK.csv output/potential/std-S3-DK.csv output/potential/train-time-S3-DK output/potential/test-time-S3-DK output/potential/tree-S3-DK.svg

output/potential/mean-S3-DK.csv output/potential/std-S3-DK.csv output/potential/train-time-S3-DK output/potential/test-time-S3-DK output/potential/tree-S3-DK.svg &: settings/potential/S3-DK.toml potential_train/feature-S3-DK.csv potential_train/target-S3-DK.csv potential_test/feature-S3-DK.csv
	./run_model.py --config=settings/potential/S3-DK.toml --train-feature=potential_train/feature-S3-DK.csv --train-target=potential_train/target-S3-DK.csv --test-feature=potential_test/feature-S3-DK.csv --predict-mean=output/potential/mean-S3-DK.csv --predict-std=output/potential/std-S3-DK.csv --visualize-tree=output/potential/tree-S3-DK.svg --train-time=output/potential/train-time-S3-DK --test-time=output/potential/test-time-S3-DK 

.PHONY: show-potential-S3-DK
show-potential-S3-DK : output/potential/mean-S3-DK.csv output/potential/std-S3-DK.csv potential_test/target-S3-DK.csv
	./plot_prediction.py --mean=output/potential/mean-S3-DK.csv --std=output/potential/std-S3-DK.csv --target=potential_test/target-S3-DK.csv 

.PHONY: run-potential-S3-FR
run-potential-S3-FR : output/potential/mean-S3-FR.csv output/potential/std-S3-FR.csv output/potential/train-time-S3-FR output/potential/test-time-S3-FR output/potential/tree-S3-FR.svg

all : output/potential/mean-S3-FR.csv output/potential/std-S3-FR.csv output/potential/train-time-S3-FR output/potential/test-time-S3-FR output/potential/tree-S3-FR.svg

output/potential/mean-S3-FR.csv output/potential/std-S3-FR.csv output/potential/train-time-S3-FR output/potential/test-time-S3-FR output/potential/tree-S3-FR.svg &: settings/potential/S3-FR.toml potential_train/feature-S3-FR.csv potential_train/target-S3-FR.csv potential_test/feature-S3-FR.csv
	./run_model.py --config=settings/potential/S3-FR.toml --train-feature=potential_train/feature-S3-FR.csv --train-target=potential_train/target-S3-FR.csv --test-feature=potential_test/feature-S3-FR.csv --predict-mean=output/potential/mean-S3-FR.csv --predict-std=output/potential/std-S3-FR.csv --visualize-tree=output/potential/tree-S3-FR.svg --train-time=output/potential/train-time-S3-FR --test-time=output/potential/test-time-S3-FR 

.PHONY: show-potential-S3-FR
show-potential-S3-FR : output/potential/mean-S3-FR.csv output/potential/std-S3-FR.csv potential_test/target-S3-FR.csv
	./plot_prediction.py --mean=output/potential/mean-S3-FR.csv --std=output/potential/std-S3-FR.csv --target=potential_test/target-S3-FR.csv 

.PHONY: run-potential-S3-BE
run-potential-S3-BE : output/potential/mean-S3-BE.csv output/potential/std-S3-BE.csv output/potential/train-time-S3-BE output/potential/test-time-S3-BE output/potential/tree-S3-BE.svg

all : output/potential/mean-S3-BE.csv output/potential/std-S3-BE.csv output/potential/train-time-S3-BE output/potential/test-time-S3-BE output/potential/tree-S3-BE.svg

output/potential/mean-S3-BE.csv output/potential/std-S3-BE.csv output/potential/train-time-S3-BE output/potential/test-time-S3-BE output/potential/tree-S3-BE.svg &: settings/potential/S3-BE.toml potential_train/feature-S3-BE.csv potential_train/target-S3-BE.csv potential_test/feature-S3-BE.csv
	./run_model.py --config=settings/potential/S3-BE.toml --train-feature=potential_train/feature-S3-BE.csv --train-target=potential_train/target-S3-BE.csv --test-feature=potential_test/feature-S3-BE.csv --predict-mean=output/potential/mean-S3-BE.csv --predict-std=output/potential/std-S3-BE.csv --visualize-tree=output/potential/tree-S3-BE.svg --train-time=output/potential/train-time-S3-BE --test-time=output/potential/test-time-S3-BE 

.PHONY: show-potential-S3-BE
show-potential-S3-BE : output/potential/mean-S3-BE.csv output/potential/std-S3-BE.csv potential_test/target-S3-BE.csv
	./plot_prediction.py --mean=output/potential/mean-S3-BE.csv --std=output/potential/std-S3-BE.csv --target=potential_test/target-S3-BE.csv 

.PHONY: run-potential-S4-WEST
run-potential-S4-WEST : output/potential/mean-S4-WEST.csv output/potential/std-S4-WEST.csv output/potential/train-time-S4-WEST output/potential/test-time-S4-WEST output/potential/tree-S4-WEST.svg

all : output/potential/mean-S4-WEST.csv output/potential/std-S4-WEST.csv output/potential/train-time-S4-WEST output/potential/test-time-S4-WEST output/potential/tree-S4-WEST.svg

output/potential/mean-S4-WEST.csv output/potential/std-S4-WEST.csv output/potential/train-time-S4-WEST output/potential/test-time-S4-WEST output/potential/tree-S4-WEST.svg &: settings/potential/S4-WEST.toml potential_train/feature-S4-WEST.csv potential_train/target-S4-WEST.csv potential_test/feature-S4-WEST.csv
	./run_model.py --config=settings/potential/S4-WEST.toml --train-feature=potential_train/feature-S4-WEST.csv --train-target=potential_train/target-S4-WEST.csv --test-feature=potential_test/feature-S4-WEST.csv --predict-mean=output/potential/mean-S4-WEST.csv --predict-std=output/potential/std-S4-WEST.csv --visualize-tree=output/potential/tree-S4-WEST.svg --train-time=output/potential/train-time-S4-WEST --test-time=output/potential/test-time-S4-WEST 

.PHONY: show-potential-S4-WEST
show-potential-S4-WEST : output/potential/mean-S4-WEST.csv output/potential/std-S4-WEST.csv potential_test/target-S4-WEST.csv
	./plot_prediction.py --mean=output/potential/mean-S4-WEST.csv --std=output/potential/std-S4-WEST.csv --target=potential_test/target-S4-WEST.csv 

.PHONY: run-potential-S2
run-potential-S2 : output/potential/mean-S2.csv output/potential/std-S2.csv output/potential/train-time-S2 output/potential/test-time-S2 output/potential/tree-S2.svg

all : output/potential/mean-S2.csv output/potential/std-S2.csv output/potential/train-time-S2 output/potential/test-time-S2 output/potential/tree-S2.svg

output/potential/mean-S2.csv output/potential/std-S2.csv output/potential/train-time-S2 output/potential/test-time-S2 output/potential/tree-S2.svg &: settings/potential/S2.toml potential_train/feature-S2.csv potential_train/target-S2.csv potential_test/feature-S2.csv
	./run_model.py --config=settings/potential/S2.toml --train-feature=potential_train/feature-S2.csv --train-target=potential_train/target-S2.csv --test-feature=potential_test/feature-S2.csv --predict-mean=output/potential/mean-S2.csv --predict-std=output/potential/std-S2.csv --visualize-tree=output/potential/tree-S2.svg --train-time=output/potential/train-time-S2 --test-time=output/potential/test-time-S2 

.PHONY: show-potential-S2
show-potential-S2 : output/potential/mean-S2.csv output/potential/std-S2.csv potential_test/target-S2.csv
	./plot_prediction.py --mean=output/potential/mean-S2.csv --std=output/potential/std-S2.csv --target=potential_test/target-S2.csv 

.PHONY: run-potential-S3-CZ
run-potential-S3-CZ : output/potential/mean-S3-CZ.csv output/potential/std-S3-CZ.csv output/potential/train-time-S3-CZ output/potential/test-time-S3-CZ output/potential/tree-S3-CZ.svg

all : output/potential/mean-S3-CZ.csv output/potential/std-S3-CZ.csv output/potential/train-time-S3-CZ output/potential/test-time-S3-CZ output/potential/tree-S3-CZ.svg

output/potential/mean-S3-CZ.csv output/potential/std-S3-CZ.csv output/potential/train-time-S3-CZ output/potential/test-time-S3-CZ output/potential/tree-S3-CZ.svg &: settings/potential/S3-CZ.toml potential_train/feature-S3-CZ.csv potential_train/target-S3-CZ.csv potential_test/feature-S3-CZ.csv
	./run_model.py --config=settings/potential/S3-CZ.toml --train-feature=potential_train/feature-S3-CZ.csv --train-target=potential_train/target-S3-CZ.csv --test-feature=potential_test/feature-S3-CZ.csv --predict-mean=output/potential/mean-S3-CZ.csv --predict-std=output/potential/std-S3-CZ.csv --visualize-tree=output/potential/tree-S3-CZ.svg --train-time=output/potential/train-time-S3-CZ --test-time=output/potential/test-time-S3-CZ 

.PHONY: show-potential-S3-CZ
show-potential-S3-CZ : output/potential/mean-S3-CZ.csv output/potential/std-S3-CZ.csv potential_test/target-S3-CZ.csv
	./plot_prediction.py --mean=output/potential/mean-S3-CZ.csv --std=output/potential/std-S3-CZ.csv --target=potential_test/target-S3-CZ.csv 

.PHONY: run-potential-S4-SOUTH
run-potential-S4-SOUTH : output/potential/mean-S4-SOUTH.csv output/potential/std-S4-SOUTH.csv output/potential/train-time-S4-SOUTH output/potential/test-time-S4-SOUTH output/potential/tree-S4-SOUTH.svg

all : output/potential/mean-S4-SOUTH.csv output/potential/std-S4-SOUTH.csv output/potential/train-time-S4-SOUTH output/potential/test-time-S4-SOUTH output/potential/tree-S4-SOUTH.svg

output/potential/mean-S4-SOUTH.csv output/potential/std-S4-SOUTH.csv output/potential/train-time-S4-SOUTH output/potential/test-time-S4-SOUTH output/potential/tree-S4-SOUTH.svg &: settings/potential/S4-SOUTH.toml potential_train/feature-S4-SOUTH.csv potential_train/target-S4-SOUTH.csv potential_test/feature-S4-SOUTH.csv
	./run_model.py --config=settings/potential/S4-SOUTH.toml --train-feature=potential_train/feature-S4-SOUTH.csv --train-target=potential_train/target-S4-SOUTH.csv --test-feature=potential_test/feature-S4-SOUTH.csv --predict-mean=output/potential/mean-S4-SOUTH.csv --predict-std=output/potential/std-S4-SOUTH.csv --visualize-tree=output/potential/tree-S4-SOUTH.svg --train-time=output/potential/train-time-S4-SOUTH --test-time=output/potential/test-time-S4-SOUTH 

.PHONY: show-potential-S4-SOUTH
show-potential-S4-SOUTH : output/potential/mean-S4-SOUTH.csv output/potential/std-S4-SOUTH.csv potential_test/target-S4-SOUTH.csv
	./plot_prediction.py --mean=output/potential/mean-S4-SOUTH.csv --std=output/potential/std-S4-SOUTH.csv --target=potential_test/target-S4-SOUTH.csv 

.PHONY: run-potential_only-S3-BG
run-potential_only-S3-BG : output/potential_only/mean-S3-BG.csv output/potential_only/std-S3-BG.csv output/potential_only/train-time-S3-BG output/potential_only/test-time-S3-BG output/potential_only/tree-S3-BG.svg

all : output/potential_only/mean-S3-BG.csv output/potential_only/std-S3-BG.csv output/potential_only/train-time-S3-BG output/potential_only/test-time-S3-BG output/potential_only/tree-S3-BG.svg

output/potential_only/mean-S3-BG.csv output/potential_only/std-S3-BG.csv output/potential_only/train-time-S3-BG output/potential_only/test-time-S3-BG output/potential_only/tree-S3-BG.svg &: settings/potential_only/S3-BG.toml potential_only_train/feature-S3-BG.csv potential_only_train/target-S3-BG.csv potential_only_test/feature-S3-BG.csv
	./run_model.py --config=settings/potential_only/S3-BG.toml --train-feature=potential_only_train/feature-S3-BG.csv --train-target=potential_only_train/target-S3-BG.csv --test-feature=potential_only_test/feature-S3-BG.csv --predict-mean=output/potential_only/mean-S3-BG.csv --predict-std=output/potential_only/std-S3-BG.csv --visualize-tree=output/potential_only/tree-S3-BG.svg --train-time=output/potential_only/train-time-S3-BG --test-time=output/potential_only/test-time-S3-BG 

.PHONY: show-potential_only-S3-BG
show-potential_only-S3-BG : output/potential_only/mean-S3-BG.csv output/potential_only/std-S3-BG.csv potential_only_test/target-S3-BG.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-BG.csv --std=output/potential_only/std-S3-BG.csv --target=potential_only_test/target-S3-BG.csv 

.PHONY: run-potential_only-S3-NL
run-potential_only-S3-NL : output/potential_only/mean-S3-NL.csv output/potential_only/std-S3-NL.csv output/potential_only/train-time-S3-NL output/potential_only/test-time-S3-NL output/potential_only/tree-S3-NL.svg

all : output/potential_only/mean-S3-NL.csv output/potential_only/std-S3-NL.csv output/potential_only/train-time-S3-NL output/potential_only/test-time-S3-NL output/potential_only/tree-S3-NL.svg

output/potential_only/mean-S3-NL.csv output/potential_only/std-S3-NL.csv output/potential_only/train-time-S3-NL output/potential_only/test-time-S3-NL output/potential_only/tree-S3-NL.svg &: settings/potential_only/S3-NL.toml potential_only_train/feature-S3-NL.csv potential_only_train/target-S3-NL.csv potential_only_test/feature-S3-NL.csv
	./run_model.py --config=settings/potential_only/S3-NL.toml --train-feature=potential_only_train/feature-S3-NL.csv --train-target=potential_only_train/target-S3-NL.csv --test-feature=potential_only_test/feature-S3-NL.csv --predict-mean=output/potential_only/mean-S3-NL.csv --predict-std=output/potential_only/std-S3-NL.csv --visualize-tree=output/potential_only/tree-S3-NL.svg --train-time=output/potential_only/train-time-S3-NL --test-time=output/potential_only/test-time-S3-NL 

.PHONY: show-potential_only-S3-NL
show-potential_only-S3-NL : output/potential_only/mean-S3-NL.csv output/potential_only/std-S3-NL.csv potential_only_test/target-S3-NL.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-NL.csv --std=output/potential_only/std-S3-NL.csv --target=potential_only_test/target-S3-NL.csv 

.PHONY: run-potential_only-S3-AT
run-potential_only-S3-AT : output/potential_only/mean-S3-AT.csv output/potential_only/std-S3-AT.csv output/potential_only/train-time-S3-AT output/potential_only/test-time-S3-AT output/potential_only/tree-S3-AT.svg

all : output/potential_only/mean-S3-AT.csv output/potential_only/std-S3-AT.csv output/potential_only/train-time-S3-AT output/potential_only/test-time-S3-AT output/potential_only/tree-S3-AT.svg

output/potential_only/mean-S3-AT.csv output/potential_only/std-S3-AT.csv output/potential_only/train-time-S3-AT output/potential_only/test-time-S3-AT output/potential_only/tree-S3-AT.svg &: settings/potential_only/S3-AT.toml potential_only_train/feature-S3-AT.csv potential_only_train/target-S3-AT.csv potential_only_test/feature-S3-AT.csv
	./run_model.py --config=settings/potential_only/S3-AT.toml --train-feature=potential_only_train/feature-S3-AT.csv --train-target=potential_only_train/target-S3-AT.csv --test-feature=potential_only_test/feature-S3-AT.csv --predict-mean=output/potential_only/mean-S3-AT.csv --predict-std=output/potential_only/std-S3-AT.csv --visualize-tree=output/potential_only/tree-S3-AT.svg --train-time=output/potential_only/train-time-S3-AT --test-time=output/potential_only/test-time-S3-AT 

.PHONY: show-potential_only-S3-AT
show-potential_only-S3-AT : output/potential_only/mean-S3-AT.csv output/potential_only/std-S3-AT.csv potential_only_test/target-S3-AT.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-AT.csv --std=output/potential_only/std-S3-AT.csv --target=potential_only_test/target-S3-AT.csv 

.PHONY: run-potential_only-S3-ES
run-potential_only-S3-ES : output/potential_only/mean-S3-ES.csv output/potential_only/std-S3-ES.csv output/potential_only/train-time-S3-ES output/potential_only/test-time-S3-ES output/potential_only/tree-S3-ES.svg

all : output/potential_only/mean-S3-ES.csv output/potential_only/std-S3-ES.csv output/potential_only/train-time-S3-ES output/potential_only/test-time-S3-ES output/potential_only/tree-S3-ES.svg

output/potential_only/mean-S3-ES.csv output/potential_only/std-S3-ES.csv output/potential_only/train-time-S3-ES output/potential_only/test-time-S3-ES output/potential_only/tree-S3-ES.svg &: settings/potential_only/S3-ES.toml potential_only_train/feature-S3-ES.csv potential_only_train/target-S3-ES.csv potential_only_test/feature-S3-ES.csv
	./run_model.py --config=settings/potential_only/S3-ES.toml --train-feature=potential_only_train/feature-S3-ES.csv --train-target=potential_only_train/target-S3-ES.csv --test-feature=potential_only_test/feature-S3-ES.csv --predict-mean=output/potential_only/mean-S3-ES.csv --predict-std=output/potential_only/std-S3-ES.csv --visualize-tree=output/potential_only/tree-S3-ES.svg --train-time=output/potential_only/train-time-S3-ES --test-time=output/potential_only/test-time-S3-ES 

.PHONY: show-potential_only-S3-ES
show-potential_only-S3-ES : output/potential_only/mean-S3-ES.csv output/potential_only/std-S3-ES.csv potential_only_test/target-S3-ES.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-ES.csv --std=output/potential_only/std-S3-ES.csv --target=potential_only_test/target-S3-ES.csv 

.PHONY: run-potential_only-S3-GR
run-potential_only-S3-GR : output/potential_only/mean-S3-GR.csv output/potential_only/std-S3-GR.csv output/potential_only/train-time-S3-GR output/potential_only/test-time-S3-GR output/potential_only/tree-S3-GR.svg

all : output/potential_only/mean-S3-GR.csv output/potential_only/std-S3-GR.csv output/potential_only/train-time-S3-GR output/potential_only/test-time-S3-GR output/potential_only/tree-S3-GR.svg

output/potential_only/mean-S3-GR.csv output/potential_only/std-S3-GR.csv output/potential_only/train-time-S3-GR output/potential_only/test-time-S3-GR output/potential_only/tree-S3-GR.svg &: settings/potential_only/S3-GR.toml potential_only_train/feature-S3-GR.csv potential_only_train/target-S3-GR.csv potential_only_test/feature-S3-GR.csv
	./run_model.py --config=settings/potential_only/S3-GR.toml --train-feature=potential_only_train/feature-S3-GR.csv --train-target=potential_only_train/target-S3-GR.csv --test-feature=potential_only_test/feature-S3-GR.csv --predict-mean=output/potential_only/mean-S3-GR.csv --predict-std=output/potential_only/std-S3-GR.csv --visualize-tree=output/potential_only/tree-S3-GR.svg --train-time=output/potential_only/train-time-S3-GR --test-time=output/potential_only/test-time-S3-GR 

.PHONY: show-potential_only-S3-GR
show-potential_only-S3-GR : output/potential_only/mean-S3-GR.csv output/potential_only/std-S3-GR.csv potential_only_test/target-S3-GR.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-GR.csv --std=output/potential_only/std-S3-GR.csv --target=potential_only_test/target-S3-GR.csv 

.PHONY: run-potential_only-S3-IT
run-potential_only-S3-IT : output/potential_only/mean-S3-IT.csv output/potential_only/std-S3-IT.csv output/potential_only/train-time-S3-IT output/potential_only/test-time-S3-IT output/potential_only/tree-S3-IT.svg

all : output/potential_only/mean-S3-IT.csv output/potential_only/std-S3-IT.csv output/potential_only/train-time-S3-IT output/potential_only/test-time-S3-IT output/potential_only/tree-S3-IT.svg

output/potential_only/mean-S3-IT.csv output/potential_only/std-S3-IT.csv output/potential_only/train-time-S3-IT output/potential_only/test-time-S3-IT output/potential_only/tree-S3-IT.svg &: settings/potential_only/S3-IT.toml potential_only_train/feature-S3-IT.csv potential_only_train/target-S3-IT.csv potential_only_test/feature-S3-IT.csv
	./run_model.py --config=settings/potential_only/S3-IT.toml --train-feature=potential_only_train/feature-S3-IT.csv --train-target=potential_only_train/target-S3-IT.csv --test-feature=potential_only_test/feature-S3-IT.csv --predict-mean=output/potential_only/mean-S3-IT.csv --predict-std=output/potential_only/std-S3-IT.csv --visualize-tree=output/potential_only/tree-S3-IT.svg --train-time=output/potential_only/train-time-S3-IT --test-time=output/potential_only/test-time-S3-IT 

.PHONY: show-potential_only-S3-IT
show-potential_only-S3-IT : output/potential_only/mean-S3-IT.csv output/potential_only/std-S3-IT.csv potential_only_test/target-S3-IT.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-IT.csv --std=output/potential_only/std-S3-IT.csv --target=potential_only_test/target-S3-IT.csv 

.PHONY: run-potential_only-S3-SI
run-potential_only-S3-SI : output/potential_only/mean-S3-SI.csv output/potential_only/std-S3-SI.csv output/potential_only/train-time-S3-SI output/potential_only/test-time-S3-SI output/potential_only/tree-S3-SI.svg

all : output/potential_only/mean-S3-SI.csv output/potential_only/std-S3-SI.csv output/potential_only/train-time-S3-SI output/potential_only/test-time-S3-SI output/potential_only/tree-S3-SI.svg

output/potential_only/mean-S3-SI.csv output/potential_only/std-S3-SI.csv output/potential_only/train-time-S3-SI output/potential_only/test-time-S3-SI output/potential_only/tree-S3-SI.svg &: settings/potential_only/S3-SI.toml potential_only_train/feature-S3-SI.csv potential_only_train/target-S3-SI.csv potential_only_test/feature-S3-SI.csv
	./run_model.py --config=settings/potential_only/S3-SI.toml --train-feature=potential_only_train/feature-S3-SI.csv --train-target=potential_only_train/target-S3-SI.csv --test-feature=potential_only_test/feature-S3-SI.csv --predict-mean=output/potential_only/mean-S3-SI.csv --predict-std=output/potential_only/std-S3-SI.csv --visualize-tree=output/potential_only/tree-S3-SI.svg --train-time=output/potential_only/train-time-S3-SI --test-time=output/potential_only/test-time-S3-SI 

.PHONY: show-potential_only-S3-SI
show-potential_only-S3-SI : output/potential_only/mean-S3-SI.csv output/potential_only/std-S3-SI.csv potential_only_test/target-S3-SI.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-SI.csv --std=output/potential_only/std-S3-SI.csv --target=potential_only_test/target-S3-SI.csv 

.PHONY: run-potential_only-S4-MIDATL
run-potential_only-S4-MIDATL : output/potential_only/mean-S4-MIDATL.csv output/potential_only/std-S4-MIDATL.csv output/potential_only/train-time-S4-MIDATL output/potential_only/test-time-S4-MIDATL output/potential_only/tree-S4-MIDATL.svg

all : output/potential_only/mean-S4-MIDATL.csv output/potential_only/std-S4-MIDATL.csv output/potential_only/train-time-S4-MIDATL output/potential_only/test-time-S4-MIDATL output/potential_only/tree-S4-MIDATL.svg

output/potential_only/mean-S4-MIDATL.csv output/potential_only/std-S4-MIDATL.csv output/potential_only/train-time-S4-MIDATL output/potential_only/test-time-S4-MIDATL output/potential_only/tree-S4-MIDATL.svg &: settings/potential_only/S4-MIDATL.toml potential_only_train/feature-S4-MIDATL.csv potential_only_train/target-S4-MIDATL.csv potential_only_test/feature-S4-MIDATL.csv
	./run_model.py --config=settings/potential_only/S4-MIDATL.toml --train-feature=potential_only_train/feature-S4-MIDATL.csv --train-target=potential_only_train/target-S4-MIDATL.csv --test-feature=potential_only_test/feature-S4-MIDATL.csv --predict-mean=output/potential_only/mean-S4-MIDATL.csv --predict-std=output/potential_only/std-S4-MIDATL.csv --visualize-tree=output/potential_only/tree-S4-MIDATL.svg --train-time=output/potential_only/train-time-S4-MIDATL --test-time=output/potential_only/test-time-S4-MIDATL 

.PHONY: show-potential_only-S4-MIDATL
show-potential_only-S4-MIDATL : output/potential_only/mean-S4-MIDATL.csv output/potential_only/std-S4-MIDATL.csv potential_only_test/target-S4-MIDATL.csv
	./plot_prediction.py --mean=output/potential_only/mean-S4-MIDATL.csv --std=output/potential_only/std-S4-MIDATL.csv --target=potential_only_test/target-S4-MIDATL.csv 

.PHONY: run-potential_only-S3-PT
run-potential_only-S3-PT : output/potential_only/mean-S3-PT.csv output/potential_only/std-S3-PT.csv output/potential_only/train-time-S3-PT output/potential_only/test-time-S3-PT output/potential_only/tree-S3-PT.svg

all : output/potential_only/mean-S3-PT.csv output/potential_only/std-S3-PT.csv output/potential_only/train-time-S3-PT output/potential_only/test-time-S3-PT output/potential_only/tree-S3-PT.svg

output/potential_only/mean-S3-PT.csv output/potential_only/std-S3-PT.csv output/potential_only/train-time-S3-PT output/potential_only/test-time-S3-PT output/potential_only/tree-S3-PT.svg &: settings/potential_only/S3-PT.toml potential_only_train/feature-S3-PT.csv potential_only_train/target-S3-PT.csv potential_only_test/feature-S3-PT.csv
	./run_model.py --config=settings/potential_only/S3-PT.toml --train-feature=potential_only_train/feature-S3-PT.csv --train-target=potential_only_train/target-S3-PT.csv --test-feature=potential_only_test/feature-S3-PT.csv --predict-mean=output/potential_only/mean-S3-PT.csv --predict-std=output/potential_only/std-S3-PT.csv --visualize-tree=output/potential_only/tree-S3-PT.svg --train-time=output/potential_only/train-time-S3-PT --test-time=output/potential_only/test-time-S3-PT 

.PHONY: show-potential_only-S3-PT
show-potential_only-S3-PT : output/potential_only/mean-S3-PT.csv output/potential_only/std-S3-PT.csv potential_only_test/target-S3-PT.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-PT.csv --std=output/potential_only/std-S3-PT.csv --target=potential_only_test/target-S3-PT.csv 

.PHONY: run-potential_only-S1
run-potential_only-S1 : output/potential_only/mean-S1.csv output/potential_only/std-S1.csv output/potential_only/train-time-S1 output/potential_only/test-time-S1 output/potential_only/tree-S1.svg

all : output/potential_only/mean-S1.csv output/potential_only/std-S1.csv output/potential_only/train-time-S1 output/potential_only/test-time-S1 output/potential_only/tree-S1.svg

output/potential_only/mean-S1.csv output/potential_only/std-S1.csv output/potential_only/train-time-S1 output/potential_only/test-time-S1 output/potential_only/tree-S1.svg &: settings/potential_only/S1.toml potential_only_train/feature-S1.csv potential_only_train/target-S1.csv potential_only_test/feature-S1.csv
	./run_model.py --config=settings/potential_only/S1.toml --train-feature=potential_only_train/feature-S1.csv --train-target=potential_only_train/target-S1.csv --test-feature=potential_only_test/feature-S1.csv --predict-mean=output/potential_only/mean-S1.csv --predict-std=output/potential_only/std-S1.csv --visualize-tree=output/potential_only/tree-S1.svg --train-time=output/potential_only/train-time-S1 --test-time=output/potential_only/test-time-S1 

.PHONY: show-potential_only-S1
show-potential_only-S1 : output/potential_only/mean-S1.csv output/potential_only/std-S1.csv potential_only_test/target-S1.csv
	./plot_prediction.py --mean=output/potential_only/mean-S1.csv --std=output/potential_only/std-S1.csv --target=potential_only_test/target-S1.csv 

.PHONY: run-potential_only-S3-CH
run-potential_only-S3-CH : output/potential_only/mean-S3-CH.csv output/potential_only/std-S3-CH.csv output/potential_only/train-time-S3-CH output/potential_only/test-time-S3-CH output/potential_only/tree-S3-CH.svg

all : output/potential_only/mean-S3-CH.csv output/potential_only/std-S3-CH.csv output/potential_only/train-time-S3-CH output/potential_only/test-time-S3-CH output/potential_only/tree-S3-CH.svg

output/potential_only/mean-S3-CH.csv output/potential_only/std-S3-CH.csv output/potential_only/train-time-S3-CH output/potential_only/test-time-S3-CH output/potential_only/tree-S3-CH.svg &: settings/potential_only/S3-CH.toml potential_only_train/feature-S3-CH.csv potential_only_train/target-S3-CH.csv potential_only_test/feature-S3-CH.csv
	./run_model.py --config=settings/potential_only/S3-CH.toml --train-feature=potential_only_train/feature-S3-CH.csv --train-target=potential_only_train/target-S3-CH.csv --test-feature=potential_only_test/feature-S3-CH.csv --predict-mean=output/potential_only/mean-S3-CH.csv --predict-std=output/potential_only/std-S3-CH.csv --visualize-tree=output/potential_only/tree-S3-CH.svg --train-time=output/potential_only/train-time-S3-CH --test-time=output/potential_only/test-time-S3-CH 

.PHONY: show-potential_only-S3-CH
show-potential_only-S3-CH : output/potential_only/mean-S3-CH.csv output/potential_only/std-S3-CH.csv potential_only_test/target-S3-CH.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-CH.csv --std=output/potential_only/std-S3-CH.csv --target=potential_only_test/target-S3-CH.csv 

.PHONY: run-potential_only-S3-SK
run-potential_only-S3-SK : output/potential_only/mean-S3-SK.csv output/potential_only/std-S3-SK.csv output/potential_only/train-time-S3-SK output/potential_only/test-time-S3-SK output/potential_only/tree-S3-SK.svg

all : output/potential_only/mean-S3-SK.csv output/potential_only/std-S3-SK.csv output/potential_only/train-time-S3-SK output/potential_only/test-time-S3-SK output/potential_only/tree-S3-SK.svg

output/potential_only/mean-S3-SK.csv output/potential_only/std-S3-SK.csv output/potential_only/train-time-S3-SK output/potential_only/test-time-S3-SK output/potential_only/tree-S3-SK.svg &: settings/potential_only/S3-SK.toml potential_only_train/feature-S3-SK.csv potential_only_train/target-S3-SK.csv potential_only_test/feature-S3-SK.csv
	./run_model.py --config=settings/potential_only/S3-SK.toml --train-feature=potential_only_train/feature-S3-SK.csv --train-target=potential_only_train/target-S3-SK.csv --test-feature=potential_only_test/feature-S3-SK.csv --predict-mean=output/potential_only/mean-S3-SK.csv --predict-std=output/potential_only/std-S3-SK.csv --visualize-tree=output/potential_only/tree-S3-SK.svg --train-time=output/potential_only/train-time-S3-SK --test-time=output/potential_only/test-time-S3-SK 

.PHONY: show-potential_only-S3-SK
show-potential_only-S3-SK : output/potential_only/mean-S3-SK.csv output/potential_only/std-S3-SK.csv potential_only_test/target-S3-SK.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-SK.csv --std=output/potential_only/std-S3-SK.csv --target=potential_only_test/target-S3-SK.csv 

.PHONY: run-potential_only-S3-DK
run-potential_only-S3-DK : output/potential_only/mean-S3-DK.csv output/potential_only/std-S3-DK.csv output/potential_only/train-time-S3-DK output/potential_only/test-time-S3-DK output/potential_only/tree-S3-DK.svg

all : output/potential_only/mean-S3-DK.csv output/potential_only/std-S3-DK.csv output/potential_only/train-time-S3-DK output/potential_only/test-time-S3-DK output/potential_only/tree-S3-DK.svg

output/potential_only/mean-S3-DK.csv output/potential_only/std-S3-DK.csv output/potential_only/train-time-S3-DK output/potential_only/test-time-S3-DK output/potential_only/tree-S3-DK.svg &: settings/potential_only/S3-DK.toml potential_only_train/feature-S3-DK.csv potential_only_train/target-S3-DK.csv potential_only_test/feature-S3-DK.csv
	./run_model.py --config=settings/potential_only/S3-DK.toml --train-feature=potential_only_train/feature-S3-DK.csv --train-target=potential_only_train/target-S3-DK.csv --test-feature=potential_only_test/feature-S3-DK.csv --predict-mean=output/potential_only/mean-S3-DK.csv --predict-std=output/potential_only/std-S3-DK.csv --visualize-tree=output/potential_only/tree-S3-DK.svg --train-time=output/potential_only/train-time-S3-DK --test-time=output/potential_only/test-time-S3-DK 

.PHONY: show-potential_only-S3-DK
show-potential_only-S3-DK : output/potential_only/mean-S3-DK.csv output/potential_only/std-S3-DK.csv potential_only_test/target-S3-DK.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-DK.csv --std=output/potential_only/std-S3-DK.csv --target=potential_only_test/target-S3-DK.csv 

.PHONY: run-potential_only-S3-FR
run-potential_only-S3-FR : output/potential_only/mean-S3-FR.csv output/potential_only/std-S3-FR.csv output/potential_only/train-time-S3-FR output/potential_only/test-time-S3-FR output/potential_only/tree-S3-FR.svg

all : output/potential_only/mean-S3-FR.csv output/potential_only/std-S3-FR.csv output/potential_only/train-time-S3-FR output/potential_only/test-time-S3-FR output/potential_only/tree-S3-FR.svg

output/potential_only/mean-S3-FR.csv output/potential_only/std-S3-FR.csv output/potential_only/train-time-S3-FR output/potential_only/test-time-S3-FR output/potential_only/tree-S3-FR.svg &: settings/potential_only/S3-FR.toml potential_only_train/feature-S3-FR.csv potential_only_train/target-S3-FR.csv potential_only_test/feature-S3-FR.csv
	./run_model.py --config=settings/potential_only/S3-FR.toml --train-feature=potential_only_train/feature-S3-FR.csv --train-target=potential_only_train/target-S3-FR.csv --test-feature=potential_only_test/feature-S3-FR.csv --predict-mean=output/potential_only/mean-S3-FR.csv --predict-std=output/potential_only/std-S3-FR.csv --visualize-tree=output/potential_only/tree-S3-FR.svg --train-time=output/potential_only/train-time-S3-FR --test-time=output/potential_only/test-time-S3-FR 

.PHONY: show-potential_only-S3-FR
show-potential_only-S3-FR : output/potential_only/mean-S3-FR.csv output/potential_only/std-S3-FR.csv potential_only_test/target-S3-FR.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-FR.csv --std=output/potential_only/std-S3-FR.csv --target=potential_only_test/target-S3-FR.csv 

.PHONY: run-potential_only-S3-BE
run-potential_only-S3-BE : output/potential_only/mean-S3-BE.csv output/potential_only/std-S3-BE.csv output/potential_only/train-time-S3-BE output/potential_only/test-time-S3-BE output/potential_only/tree-S3-BE.svg

all : output/potential_only/mean-S3-BE.csv output/potential_only/std-S3-BE.csv output/potential_only/train-time-S3-BE output/potential_only/test-time-S3-BE output/potential_only/tree-S3-BE.svg

output/potential_only/mean-S3-BE.csv output/potential_only/std-S3-BE.csv output/potential_only/train-time-S3-BE output/potential_only/test-time-S3-BE output/potential_only/tree-S3-BE.svg &: settings/potential_only/S3-BE.toml potential_only_train/feature-S3-BE.csv potential_only_train/target-S3-BE.csv potential_only_test/feature-S3-BE.csv
	./run_model.py --config=settings/potential_only/S3-BE.toml --train-feature=potential_only_train/feature-S3-BE.csv --train-target=potential_only_train/target-S3-BE.csv --test-feature=potential_only_test/feature-S3-BE.csv --predict-mean=output/potential_only/mean-S3-BE.csv --predict-std=output/potential_only/std-S3-BE.csv --visualize-tree=output/potential_only/tree-S3-BE.svg --train-time=output/potential_only/train-time-S3-BE --test-time=output/potential_only/test-time-S3-BE 

.PHONY: show-potential_only-S3-BE
show-potential_only-S3-BE : output/potential_only/mean-S3-BE.csv output/potential_only/std-S3-BE.csv potential_only_test/target-S3-BE.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-BE.csv --std=output/potential_only/std-S3-BE.csv --target=potential_only_test/target-S3-BE.csv 

.PHONY: run-potential_only-S4-WEST
run-potential_only-S4-WEST : output/potential_only/mean-S4-WEST.csv output/potential_only/std-S4-WEST.csv output/potential_only/train-time-S4-WEST output/potential_only/test-time-S4-WEST output/potential_only/tree-S4-WEST.svg

all : output/potential_only/mean-S4-WEST.csv output/potential_only/std-S4-WEST.csv output/potential_only/train-time-S4-WEST output/potential_only/test-time-S4-WEST output/potential_only/tree-S4-WEST.svg

output/potential_only/mean-S4-WEST.csv output/potential_only/std-S4-WEST.csv output/potential_only/train-time-S4-WEST output/potential_only/test-time-S4-WEST output/potential_only/tree-S4-WEST.svg &: settings/potential_only/S4-WEST.toml potential_only_train/feature-S4-WEST.csv potential_only_train/target-S4-WEST.csv potential_only_test/feature-S4-WEST.csv
	./run_model.py --config=settings/potential_only/S4-WEST.toml --train-feature=potential_only_train/feature-S4-WEST.csv --train-target=potential_only_train/target-S4-WEST.csv --test-feature=potential_only_test/feature-S4-WEST.csv --predict-mean=output/potential_only/mean-S4-WEST.csv --predict-std=output/potential_only/std-S4-WEST.csv --visualize-tree=output/potential_only/tree-S4-WEST.svg --train-time=output/potential_only/train-time-S4-WEST --test-time=output/potential_only/test-time-S4-WEST 

.PHONY: show-potential_only-S4-WEST
show-potential_only-S4-WEST : output/potential_only/mean-S4-WEST.csv output/potential_only/std-S4-WEST.csv potential_only_test/target-S4-WEST.csv
	./plot_prediction.py --mean=output/potential_only/mean-S4-WEST.csv --std=output/potential_only/std-S4-WEST.csv --target=potential_only_test/target-S4-WEST.csv 

.PHONY: run-potential_only-S2
run-potential_only-S2 : output/potential_only/mean-S2.csv output/potential_only/std-S2.csv output/potential_only/train-time-S2 output/potential_only/test-time-S2 output/potential_only/tree-S2.svg

all : output/potential_only/mean-S2.csv output/potential_only/std-S2.csv output/potential_only/train-time-S2 output/potential_only/test-time-S2 output/potential_only/tree-S2.svg

output/potential_only/mean-S2.csv output/potential_only/std-S2.csv output/potential_only/train-time-S2 output/potential_only/test-time-S2 output/potential_only/tree-S2.svg &: settings/potential_only/S2.toml potential_only_train/feature-S2.csv potential_only_train/target-S2.csv potential_only_test/feature-S2.csv
	./run_model.py --config=settings/potential_only/S2.toml --train-feature=potential_only_train/feature-S2.csv --train-target=potential_only_train/target-S2.csv --test-feature=potential_only_test/feature-S2.csv --predict-mean=output/potential_only/mean-S2.csv --predict-std=output/potential_only/std-S2.csv --visualize-tree=output/potential_only/tree-S2.svg --train-time=output/potential_only/train-time-S2 --test-time=output/potential_only/test-time-S2 

.PHONY: show-potential_only-S2
show-potential_only-S2 : output/potential_only/mean-S2.csv output/potential_only/std-S2.csv potential_only_test/target-S2.csv
	./plot_prediction.py --mean=output/potential_only/mean-S2.csv --std=output/potential_only/std-S2.csv --target=potential_only_test/target-S2.csv 

.PHONY: run-potential_only-S3-CZ
run-potential_only-S3-CZ : output/potential_only/mean-S3-CZ.csv output/potential_only/std-S3-CZ.csv output/potential_only/train-time-S3-CZ output/potential_only/test-time-S3-CZ output/potential_only/tree-S3-CZ.svg

all : output/potential_only/mean-S3-CZ.csv output/potential_only/std-S3-CZ.csv output/potential_only/train-time-S3-CZ output/potential_only/test-time-S3-CZ output/potential_only/tree-S3-CZ.svg

output/potential_only/mean-S3-CZ.csv output/potential_only/std-S3-CZ.csv output/potential_only/train-time-S3-CZ output/potential_only/test-time-S3-CZ output/potential_only/tree-S3-CZ.svg &: settings/potential_only/S3-CZ.toml potential_only_train/feature-S3-CZ.csv potential_only_train/target-S3-CZ.csv potential_only_test/feature-S3-CZ.csv
	./run_model.py --config=settings/potential_only/S3-CZ.toml --train-feature=potential_only_train/feature-S3-CZ.csv --train-target=potential_only_train/target-S3-CZ.csv --test-feature=potential_only_test/feature-S3-CZ.csv --predict-mean=output/potential_only/mean-S3-CZ.csv --predict-std=output/potential_only/std-S3-CZ.csv --visualize-tree=output/potential_only/tree-S3-CZ.svg --train-time=output/potential_only/train-time-S3-CZ --test-time=output/potential_only/test-time-S3-CZ 

.PHONY: show-potential_only-S3-CZ
show-potential_only-S3-CZ : output/potential_only/mean-S3-CZ.csv output/potential_only/std-S3-CZ.csv potential_only_test/target-S3-CZ.csv
	./plot_prediction.py --mean=output/potential_only/mean-S3-CZ.csv --std=output/potential_only/std-S3-CZ.csv --target=potential_only_test/target-S3-CZ.csv 

.PHONY: run-potential_only-S4-SOUTH
run-potential_only-S4-SOUTH : output/potential_only/mean-S4-SOUTH.csv output/potential_only/std-S4-SOUTH.csv output/potential_only/train-time-S4-SOUTH output/potential_only/test-time-S4-SOUTH output/potential_only/tree-S4-SOUTH.svg

all : output/potential_only/mean-S4-SOUTH.csv output/potential_only/std-S4-SOUTH.csv output/potential_only/train-time-S4-SOUTH output/potential_only/test-time-S4-SOUTH output/potential_only/tree-S4-SOUTH.svg

output/potential_only/mean-S4-SOUTH.csv output/potential_only/std-S4-SOUTH.csv output/potential_only/train-time-S4-SOUTH output/potential_only/test-time-S4-SOUTH output/potential_only/tree-S4-SOUTH.svg &: settings/potential_only/S4-SOUTH.toml potential_only_train/feature-S4-SOUTH.csv potential_only_train/target-S4-SOUTH.csv potential_only_test/feature-S4-SOUTH.csv
	./run_model.py --config=settings/potential_only/S4-SOUTH.toml --train-feature=potential_only_train/feature-S4-SOUTH.csv --train-target=potential_only_train/target-S4-SOUTH.csv --test-feature=potential_only_test/feature-S4-SOUTH.csv --predict-mean=output/potential_only/mean-S4-SOUTH.csv --predict-std=output/potential_only/std-S4-SOUTH.csv --visualize-tree=output/potential_only/tree-S4-SOUTH.svg --train-time=output/potential_only/train-time-S4-SOUTH --test-time=output/potential_only/test-time-S4-SOUTH 

.PHONY: show-potential_only-S4-SOUTH
show-potential_only-S4-SOUTH : output/potential_only/mean-S4-SOUTH.csv output/potential_only/std-S4-SOUTH.csv potential_only_test/target-S4-SOUTH.csv
	./plot_prediction.py --mean=output/potential_only/mean-S4-SOUTH.csv --std=output/potential_only/std-S4-SOUTH.csv --target=potential_only_test/target-S4-SOUTH.csv 

.PHONY: run-netload_only-S3-BG
run-netload_only-S3-BG : output/netload_only/mean-S3-BG.csv output/netload_only/std-S3-BG.csv output/netload_only/train-time-S3-BG output/netload_only/test-time-S3-BG output/netload_only/tree-S3-BG.svg

all : output/netload_only/mean-S3-BG.csv output/netload_only/std-S3-BG.csv output/netload_only/train-time-S3-BG output/netload_only/test-time-S3-BG output/netload_only/tree-S3-BG.svg

output/netload_only/mean-S3-BG.csv output/netload_only/std-S3-BG.csv output/netload_only/train-time-S3-BG output/netload_only/test-time-S3-BG output/netload_only/tree-S3-BG.svg &: settings/netload_only/S3-BG.toml netload_only_train/feature-S3-BG.csv netload_only_train/target-S3-BG.csv netload_only_test/feature-S3-BG.csv
	./run_model.py --config=settings/netload_only/S3-BG.toml --train-feature=netload_only_train/feature-S3-BG.csv --train-target=netload_only_train/target-S3-BG.csv --test-feature=netload_only_test/feature-S3-BG.csv --predict-mean=output/netload_only/mean-S3-BG.csv --predict-std=output/netload_only/std-S3-BG.csv --visualize-tree=output/netload_only/tree-S3-BG.svg --train-time=output/netload_only/train-time-S3-BG --test-time=output/netload_only/test-time-S3-BG 

.PHONY: show-netload_only-S3-BG
show-netload_only-S3-BG : output/netload_only/mean-S3-BG.csv output/netload_only/std-S3-BG.csv netload_only_test/target-S3-BG.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-BG.csv --std=output/netload_only/std-S3-BG.csv --target=netload_only_test/target-S3-BG.csv 

.PHONY: run-netload_only-S3-NL
run-netload_only-S3-NL : output/netload_only/mean-S3-NL.csv output/netload_only/std-S3-NL.csv output/netload_only/train-time-S3-NL output/netload_only/test-time-S3-NL output/netload_only/tree-S3-NL.svg

all : output/netload_only/mean-S3-NL.csv output/netload_only/std-S3-NL.csv output/netload_only/train-time-S3-NL output/netload_only/test-time-S3-NL output/netload_only/tree-S3-NL.svg

output/netload_only/mean-S3-NL.csv output/netload_only/std-S3-NL.csv output/netload_only/train-time-S3-NL output/netload_only/test-time-S3-NL output/netload_only/tree-S3-NL.svg &: settings/netload_only/S3-NL.toml netload_only_train/feature-S3-NL.csv netload_only_train/target-S3-NL.csv netload_only_test/feature-S3-NL.csv
	./run_model.py --config=settings/netload_only/S3-NL.toml --train-feature=netload_only_train/feature-S3-NL.csv --train-target=netload_only_train/target-S3-NL.csv --test-feature=netload_only_test/feature-S3-NL.csv --predict-mean=output/netload_only/mean-S3-NL.csv --predict-std=output/netload_only/std-S3-NL.csv --visualize-tree=output/netload_only/tree-S3-NL.svg --train-time=output/netload_only/train-time-S3-NL --test-time=output/netload_only/test-time-S3-NL 

.PHONY: show-netload_only-S3-NL
show-netload_only-S3-NL : output/netload_only/mean-S3-NL.csv output/netload_only/std-S3-NL.csv netload_only_test/target-S3-NL.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-NL.csv --std=output/netload_only/std-S3-NL.csv --target=netload_only_test/target-S3-NL.csv 

.PHONY: run-netload_only-S3-AT
run-netload_only-S3-AT : output/netload_only/mean-S3-AT.csv output/netload_only/std-S3-AT.csv output/netload_only/train-time-S3-AT output/netload_only/test-time-S3-AT output/netload_only/tree-S3-AT.svg

all : output/netload_only/mean-S3-AT.csv output/netload_only/std-S3-AT.csv output/netload_only/train-time-S3-AT output/netload_only/test-time-S3-AT output/netload_only/tree-S3-AT.svg

output/netload_only/mean-S3-AT.csv output/netload_only/std-S3-AT.csv output/netload_only/train-time-S3-AT output/netload_only/test-time-S3-AT output/netload_only/tree-S3-AT.svg &: settings/netload_only/S3-AT.toml netload_only_train/feature-S3-AT.csv netload_only_train/target-S3-AT.csv netload_only_test/feature-S3-AT.csv
	./run_model.py --config=settings/netload_only/S3-AT.toml --train-feature=netload_only_train/feature-S3-AT.csv --train-target=netload_only_train/target-S3-AT.csv --test-feature=netload_only_test/feature-S3-AT.csv --predict-mean=output/netload_only/mean-S3-AT.csv --predict-std=output/netload_only/std-S3-AT.csv --visualize-tree=output/netload_only/tree-S3-AT.svg --train-time=output/netload_only/train-time-S3-AT --test-time=output/netload_only/test-time-S3-AT 

.PHONY: show-netload_only-S3-AT
show-netload_only-S3-AT : output/netload_only/mean-S3-AT.csv output/netload_only/std-S3-AT.csv netload_only_test/target-S3-AT.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-AT.csv --std=output/netload_only/std-S3-AT.csv --target=netload_only_test/target-S3-AT.csv 

.PHONY: run-netload_only-S3-ES
run-netload_only-S3-ES : output/netload_only/mean-S3-ES.csv output/netload_only/std-S3-ES.csv output/netload_only/train-time-S3-ES output/netload_only/test-time-S3-ES output/netload_only/tree-S3-ES.svg

all : output/netload_only/mean-S3-ES.csv output/netload_only/std-S3-ES.csv output/netload_only/train-time-S3-ES output/netload_only/test-time-S3-ES output/netload_only/tree-S3-ES.svg

output/netload_only/mean-S3-ES.csv output/netload_only/std-S3-ES.csv output/netload_only/train-time-S3-ES output/netload_only/test-time-S3-ES output/netload_only/tree-S3-ES.svg &: settings/netload_only/S3-ES.toml netload_only_train/feature-S3-ES.csv netload_only_train/target-S3-ES.csv netload_only_test/feature-S3-ES.csv
	./run_model.py --config=settings/netload_only/S3-ES.toml --train-feature=netload_only_train/feature-S3-ES.csv --train-target=netload_only_train/target-S3-ES.csv --test-feature=netload_only_test/feature-S3-ES.csv --predict-mean=output/netload_only/mean-S3-ES.csv --predict-std=output/netload_only/std-S3-ES.csv --visualize-tree=output/netload_only/tree-S3-ES.svg --train-time=output/netload_only/train-time-S3-ES --test-time=output/netload_only/test-time-S3-ES 

.PHONY: show-netload_only-S3-ES
show-netload_only-S3-ES : output/netload_only/mean-S3-ES.csv output/netload_only/std-S3-ES.csv netload_only_test/target-S3-ES.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-ES.csv --std=output/netload_only/std-S3-ES.csv --target=netload_only_test/target-S3-ES.csv 

.PHONY: run-netload_only-S3-GR
run-netload_only-S3-GR : output/netload_only/mean-S3-GR.csv output/netload_only/std-S3-GR.csv output/netload_only/train-time-S3-GR output/netload_only/test-time-S3-GR output/netload_only/tree-S3-GR.svg

all : output/netload_only/mean-S3-GR.csv output/netload_only/std-S3-GR.csv output/netload_only/train-time-S3-GR output/netload_only/test-time-S3-GR output/netload_only/tree-S3-GR.svg

output/netload_only/mean-S3-GR.csv output/netload_only/std-S3-GR.csv output/netload_only/train-time-S3-GR output/netload_only/test-time-S3-GR output/netload_only/tree-S3-GR.svg &: settings/netload_only/S3-GR.toml netload_only_train/feature-S3-GR.csv netload_only_train/target-S3-GR.csv netload_only_test/feature-S3-GR.csv
	./run_model.py --config=settings/netload_only/S3-GR.toml --train-feature=netload_only_train/feature-S3-GR.csv --train-target=netload_only_train/target-S3-GR.csv --test-feature=netload_only_test/feature-S3-GR.csv --predict-mean=output/netload_only/mean-S3-GR.csv --predict-std=output/netload_only/std-S3-GR.csv --visualize-tree=output/netload_only/tree-S3-GR.svg --train-time=output/netload_only/train-time-S3-GR --test-time=output/netload_only/test-time-S3-GR 

.PHONY: show-netload_only-S3-GR
show-netload_only-S3-GR : output/netload_only/mean-S3-GR.csv output/netload_only/std-S3-GR.csv netload_only_test/target-S3-GR.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-GR.csv --std=output/netload_only/std-S3-GR.csv --target=netload_only_test/target-S3-GR.csv 

.PHONY: run-netload_only-S3-IT
run-netload_only-S3-IT : output/netload_only/mean-S3-IT.csv output/netload_only/std-S3-IT.csv output/netload_only/train-time-S3-IT output/netload_only/test-time-S3-IT output/netload_only/tree-S3-IT.svg

all : output/netload_only/mean-S3-IT.csv output/netload_only/std-S3-IT.csv output/netload_only/train-time-S3-IT output/netload_only/test-time-S3-IT output/netload_only/tree-S3-IT.svg

output/netload_only/mean-S3-IT.csv output/netload_only/std-S3-IT.csv output/netload_only/train-time-S3-IT output/netload_only/test-time-S3-IT output/netload_only/tree-S3-IT.svg &: settings/netload_only/S3-IT.toml netload_only_train/feature-S3-IT.csv netload_only_train/target-S3-IT.csv netload_only_test/feature-S3-IT.csv
	./run_model.py --config=settings/netload_only/S3-IT.toml --train-feature=netload_only_train/feature-S3-IT.csv --train-target=netload_only_train/target-S3-IT.csv --test-feature=netload_only_test/feature-S3-IT.csv --predict-mean=output/netload_only/mean-S3-IT.csv --predict-std=output/netload_only/std-S3-IT.csv --visualize-tree=output/netload_only/tree-S3-IT.svg --train-time=output/netload_only/train-time-S3-IT --test-time=output/netload_only/test-time-S3-IT 

.PHONY: show-netload_only-S3-IT
show-netload_only-S3-IT : output/netload_only/mean-S3-IT.csv output/netload_only/std-S3-IT.csv netload_only_test/target-S3-IT.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-IT.csv --std=output/netload_only/std-S3-IT.csv --target=netload_only_test/target-S3-IT.csv 

.PHONY: run-netload_only-S3-SI
run-netload_only-S3-SI : output/netload_only/mean-S3-SI.csv output/netload_only/std-S3-SI.csv output/netload_only/train-time-S3-SI output/netload_only/test-time-S3-SI output/netload_only/tree-S3-SI.svg

all : output/netload_only/mean-S3-SI.csv output/netload_only/std-S3-SI.csv output/netload_only/train-time-S3-SI output/netload_only/test-time-S3-SI output/netload_only/tree-S3-SI.svg

output/netload_only/mean-S3-SI.csv output/netload_only/std-S3-SI.csv output/netload_only/train-time-S3-SI output/netload_only/test-time-S3-SI output/netload_only/tree-S3-SI.svg &: settings/netload_only/S3-SI.toml netload_only_train/feature-S3-SI.csv netload_only_train/target-S3-SI.csv netload_only_test/feature-S3-SI.csv
	./run_model.py --config=settings/netload_only/S3-SI.toml --train-feature=netload_only_train/feature-S3-SI.csv --train-target=netload_only_train/target-S3-SI.csv --test-feature=netload_only_test/feature-S3-SI.csv --predict-mean=output/netload_only/mean-S3-SI.csv --predict-std=output/netload_only/std-S3-SI.csv --visualize-tree=output/netload_only/tree-S3-SI.svg --train-time=output/netload_only/train-time-S3-SI --test-time=output/netload_only/test-time-S3-SI 

.PHONY: show-netload_only-S3-SI
show-netload_only-S3-SI : output/netload_only/mean-S3-SI.csv output/netload_only/std-S3-SI.csv netload_only_test/target-S3-SI.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-SI.csv --std=output/netload_only/std-S3-SI.csv --target=netload_only_test/target-S3-SI.csv 

.PHONY: run-netload_only-S4-MIDATL
run-netload_only-S4-MIDATL : output/netload_only/mean-S4-MIDATL.csv output/netload_only/std-S4-MIDATL.csv output/netload_only/train-time-S4-MIDATL output/netload_only/test-time-S4-MIDATL output/netload_only/tree-S4-MIDATL.svg

all : output/netload_only/mean-S4-MIDATL.csv output/netload_only/std-S4-MIDATL.csv output/netload_only/train-time-S4-MIDATL output/netload_only/test-time-S4-MIDATL output/netload_only/tree-S4-MIDATL.svg

output/netload_only/mean-S4-MIDATL.csv output/netload_only/std-S4-MIDATL.csv output/netload_only/train-time-S4-MIDATL output/netload_only/test-time-S4-MIDATL output/netload_only/tree-S4-MIDATL.svg &: settings/netload_only/S4-MIDATL.toml netload_only_train/feature-S4-MIDATL.csv netload_only_train/target-S4-MIDATL.csv netload_only_test/feature-S4-MIDATL.csv
	./run_model.py --config=settings/netload_only/S4-MIDATL.toml --train-feature=netload_only_train/feature-S4-MIDATL.csv --train-target=netload_only_train/target-S4-MIDATL.csv --test-feature=netload_only_test/feature-S4-MIDATL.csv --predict-mean=output/netload_only/mean-S4-MIDATL.csv --predict-std=output/netload_only/std-S4-MIDATL.csv --visualize-tree=output/netload_only/tree-S4-MIDATL.svg --train-time=output/netload_only/train-time-S4-MIDATL --test-time=output/netload_only/test-time-S4-MIDATL 

.PHONY: show-netload_only-S4-MIDATL
show-netload_only-S4-MIDATL : output/netload_only/mean-S4-MIDATL.csv output/netload_only/std-S4-MIDATL.csv netload_only_test/target-S4-MIDATL.csv
	./plot_prediction.py --mean=output/netload_only/mean-S4-MIDATL.csv --std=output/netload_only/std-S4-MIDATL.csv --target=netload_only_test/target-S4-MIDATL.csv 

.PHONY: run-netload_only-S3-PT
run-netload_only-S3-PT : output/netload_only/mean-S3-PT.csv output/netload_only/std-S3-PT.csv output/netload_only/train-time-S3-PT output/netload_only/test-time-S3-PT output/netload_only/tree-S3-PT.svg

all : output/netload_only/mean-S3-PT.csv output/netload_only/std-S3-PT.csv output/netload_only/train-time-S3-PT output/netload_only/test-time-S3-PT output/netload_only/tree-S3-PT.svg

output/netload_only/mean-S3-PT.csv output/netload_only/std-S3-PT.csv output/netload_only/train-time-S3-PT output/netload_only/test-time-S3-PT output/netload_only/tree-S3-PT.svg &: settings/netload_only/S3-PT.toml netload_only_train/feature-S3-PT.csv netload_only_train/target-S3-PT.csv netload_only_test/feature-S3-PT.csv
	./run_model.py --config=settings/netload_only/S3-PT.toml --train-feature=netload_only_train/feature-S3-PT.csv --train-target=netload_only_train/target-S3-PT.csv --test-feature=netload_only_test/feature-S3-PT.csv --predict-mean=output/netload_only/mean-S3-PT.csv --predict-std=output/netload_only/std-S3-PT.csv --visualize-tree=output/netload_only/tree-S3-PT.svg --train-time=output/netload_only/train-time-S3-PT --test-time=output/netload_only/test-time-S3-PT 

.PHONY: show-netload_only-S3-PT
show-netload_only-S3-PT : output/netload_only/mean-S3-PT.csv output/netload_only/std-S3-PT.csv netload_only_test/target-S3-PT.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-PT.csv --std=output/netload_only/std-S3-PT.csv --target=netload_only_test/target-S3-PT.csv 

.PHONY: run-netload_only-S1
run-netload_only-S1 : output/netload_only/mean-S1.csv output/netload_only/std-S1.csv output/netload_only/train-time-S1 output/netload_only/test-time-S1 output/netload_only/tree-S1.svg

all : output/netload_only/mean-S1.csv output/netload_only/std-S1.csv output/netload_only/train-time-S1 output/netload_only/test-time-S1 output/netload_only/tree-S1.svg

output/netload_only/mean-S1.csv output/netload_only/std-S1.csv output/netload_only/train-time-S1 output/netload_only/test-time-S1 output/netload_only/tree-S1.svg &: settings/netload_only/S1.toml netload_only_train/feature-S1.csv netload_only_train/target-S1.csv netload_only_test/feature-S1.csv
	./run_model.py --config=settings/netload_only/S1.toml --train-feature=netload_only_train/feature-S1.csv --train-target=netload_only_train/target-S1.csv --test-feature=netload_only_test/feature-S1.csv --predict-mean=output/netload_only/mean-S1.csv --predict-std=output/netload_only/std-S1.csv --visualize-tree=output/netload_only/tree-S1.svg --train-time=output/netload_only/train-time-S1 --test-time=output/netload_only/test-time-S1 

.PHONY: show-netload_only-S1
show-netload_only-S1 : output/netload_only/mean-S1.csv output/netload_only/std-S1.csv netload_only_test/target-S1.csv
	./plot_prediction.py --mean=output/netload_only/mean-S1.csv --std=output/netload_only/std-S1.csv --target=netload_only_test/target-S1.csv 

.PHONY: run-netload_only-S3-CH
run-netload_only-S3-CH : output/netload_only/mean-S3-CH.csv output/netload_only/std-S3-CH.csv output/netload_only/train-time-S3-CH output/netload_only/test-time-S3-CH output/netload_only/tree-S3-CH.svg

all : output/netload_only/mean-S3-CH.csv output/netload_only/std-S3-CH.csv output/netload_only/train-time-S3-CH output/netload_only/test-time-S3-CH output/netload_only/tree-S3-CH.svg

output/netload_only/mean-S3-CH.csv output/netload_only/std-S3-CH.csv output/netload_only/train-time-S3-CH output/netload_only/test-time-S3-CH output/netload_only/tree-S3-CH.svg &: settings/netload_only/S3-CH.toml netload_only_train/feature-S3-CH.csv netload_only_train/target-S3-CH.csv netload_only_test/feature-S3-CH.csv
	./run_model.py --config=settings/netload_only/S3-CH.toml --train-feature=netload_only_train/feature-S3-CH.csv --train-target=netload_only_train/target-S3-CH.csv --test-feature=netload_only_test/feature-S3-CH.csv --predict-mean=output/netload_only/mean-S3-CH.csv --predict-std=output/netload_only/std-S3-CH.csv --visualize-tree=output/netload_only/tree-S3-CH.svg --train-time=output/netload_only/train-time-S3-CH --test-time=output/netload_only/test-time-S3-CH 

.PHONY: show-netload_only-S3-CH
show-netload_only-S3-CH : output/netload_only/mean-S3-CH.csv output/netload_only/std-S3-CH.csv netload_only_test/target-S3-CH.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-CH.csv --std=output/netload_only/std-S3-CH.csv --target=netload_only_test/target-S3-CH.csv 

.PHONY: run-netload_only-S3-SK
run-netload_only-S3-SK : output/netload_only/mean-S3-SK.csv output/netload_only/std-S3-SK.csv output/netload_only/train-time-S3-SK output/netload_only/test-time-S3-SK output/netload_only/tree-S3-SK.svg

all : output/netload_only/mean-S3-SK.csv output/netload_only/std-S3-SK.csv output/netload_only/train-time-S3-SK output/netload_only/test-time-S3-SK output/netload_only/tree-S3-SK.svg

output/netload_only/mean-S3-SK.csv output/netload_only/std-S3-SK.csv output/netload_only/train-time-S3-SK output/netload_only/test-time-S3-SK output/netload_only/tree-S3-SK.svg &: settings/netload_only/S3-SK.toml netload_only_train/feature-S3-SK.csv netload_only_train/target-S3-SK.csv netload_only_test/feature-S3-SK.csv
	./run_model.py --config=settings/netload_only/S3-SK.toml --train-feature=netload_only_train/feature-S3-SK.csv --train-target=netload_only_train/target-S3-SK.csv --test-feature=netload_only_test/feature-S3-SK.csv --predict-mean=output/netload_only/mean-S3-SK.csv --predict-std=output/netload_only/std-S3-SK.csv --visualize-tree=output/netload_only/tree-S3-SK.svg --train-time=output/netload_only/train-time-S3-SK --test-time=output/netload_only/test-time-S3-SK 

.PHONY: show-netload_only-S3-SK
show-netload_only-S3-SK : output/netload_only/mean-S3-SK.csv output/netload_only/std-S3-SK.csv netload_only_test/target-S3-SK.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-SK.csv --std=output/netload_only/std-S3-SK.csv --target=netload_only_test/target-S3-SK.csv 

.PHONY: run-netload_only-S3-DK
run-netload_only-S3-DK : output/netload_only/mean-S3-DK.csv output/netload_only/std-S3-DK.csv output/netload_only/train-time-S3-DK output/netload_only/test-time-S3-DK output/netload_only/tree-S3-DK.svg

all : output/netload_only/mean-S3-DK.csv output/netload_only/std-S3-DK.csv output/netload_only/train-time-S3-DK output/netload_only/test-time-S3-DK output/netload_only/tree-S3-DK.svg

output/netload_only/mean-S3-DK.csv output/netload_only/std-S3-DK.csv output/netload_only/train-time-S3-DK output/netload_only/test-time-S3-DK output/netload_only/tree-S3-DK.svg &: settings/netload_only/S3-DK.toml netload_only_train/feature-S3-DK.csv netload_only_train/target-S3-DK.csv netload_only_test/feature-S3-DK.csv
	./run_model.py --config=settings/netload_only/S3-DK.toml --train-feature=netload_only_train/feature-S3-DK.csv --train-target=netload_only_train/target-S3-DK.csv --test-feature=netload_only_test/feature-S3-DK.csv --predict-mean=output/netload_only/mean-S3-DK.csv --predict-std=output/netload_only/std-S3-DK.csv --visualize-tree=output/netload_only/tree-S3-DK.svg --train-time=output/netload_only/train-time-S3-DK --test-time=output/netload_only/test-time-S3-DK 

.PHONY: show-netload_only-S3-DK
show-netload_only-S3-DK : output/netload_only/mean-S3-DK.csv output/netload_only/std-S3-DK.csv netload_only_test/target-S3-DK.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-DK.csv --std=output/netload_only/std-S3-DK.csv --target=netload_only_test/target-S3-DK.csv 

.PHONY: run-netload_only-S3-FR
run-netload_only-S3-FR : output/netload_only/mean-S3-FR.csv output/netload_only/std-S3-FR.csv output/netload_only/train-time-S3-FR output/netload_only/test-time-S3-FR output/netload_only/tree-S3-FR.svg

all : output/netload_only/mean-S3-FR.csv output/netload_only/std-S3-FR.csv output/netload_only/train-time-S3-FR output/netload_only/test-time-S3-FR output/netload_only/tree-S3-FR.svg

output/netload_only/mean-S3-FR.csv output/netload_only/std-S3-FR.csv output/netload_only/train-time-S3-FR output/netload_only/test-time-S3-FR output/netload_only/tree-S3-FR.svg &: settings/netload_only/S3-FR.toml netload_only_train/feature-S3-FR.csv netload_only_train/target-S3-FR.csv netload_only_test/feature-S3-FR.csv
	./run_model.py --config=settings/netload_only/S3-FR.toml --train-feature=netload_only_train/feature-S3-FR.csv --train-target=netload_only_train/target-S3-FR.csv --test-feature=netload_only_test/feature-S3-FR.csv --predict-mean=output/netload_only/mean-S3-FR.csv --predict-std=output/netload_only/std-S3-FR.csv --visualize-tree=output/netload_only/tree-S3-FR.svg --train-time=output/netload_only/train-time-S3-FR --test-time=output/netload_only/test-time-S3-FR 

.PHONY: show-netload_only-S3-FR
show-netload_only-S3-FR : output/netload_only/mean-S3-FR.csv output/netload_only/std-S3-FR.csv netload_only_test/target-S3-FR.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-FR.csv --std=output/netload_only/std-S3-FR.csv --target=netload_only_test/target-S3-FR.csv 

.PHONY: run-netload_only-S3-BE
run-netload_only-S3-BE : output/netload_only/mean-S3-BE.csv output/netload_only/std-S3-BE.csv output/netload_only/train-time-S3-BE output/netload_only/test-time-S3-BE output/netload_only/tree-S3-BE.svg

all : output/netload_only/mean-S3-BE.csv output/netload_only/std-S3-BE.csv output/netload_only/train-time-S3-BE output/netload_only/test-time-S3-BE output/netload_only/tree-S3-BE.svg

output/netload_only/mean-S3-BE.csv output/netload_only/std-S3-BE.csv output/netload_only/train-time-S3-BE output/netload_only/test-time-S3-BE output/netload_only/tree-S3-BE.svg &: settings/netload_only/S3-BE.toml netload_only_train/feature-S3-BE.csv netload_only_train/target-S3-BE.csv netload_only_test/feature-S3-BE.csv
	./run_model.py --config=settings/netload_only/S3-BE.toml --train-feature=netload_only_train/feature-S3-BE.csv --train-target=netload_only_train/target-S3-BE.csv --test-feature=netload_only_test/feature-S3-BE.csv --predict-mean=output/netload_only/mean-S3-BE.csv --predict-std=output/netload_only/std-S3-BE.csv --visualize-tree=output/netload_only/tree-S3-BE.svg --train-time=output/netload_only/train-time-S3-BE --test-time=output/netload_only/test-time-S3-BE 

.PHONY: show-netload_only-S3-BE
show-netload_only-S3-BE : output/netload_only/mean-S3-BE.csv output/netload_only/std-S3-BE.csv netload_only_test/target-S3-BE.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-BE.csv --std=output/netload_only/std-S3-BE.csv --target=netload_only_test/target-S3-BE.csv 

.PHONY: run-netload_only-S4-WEST
run-netload_only-S4-WEST : output/netload_only/mean-S4-WEST.csv output/netload_only/std-S4-WEST.csv output/netload_only/train-time-S4-WEST output/netload_only/test-time-S4-WEST output/netload_only/tree-S4-WEST.svg

all : output/netload_only/mean-S4-WEST.csv output/netload_only/std-S4-WEST.csv output/netload_only/train-time-S4-WEST output/netload_only/test-time-S4-WEST output/netload_only/tree-S4-WEST.svg

output/netload_only/mean-S4-WEST.csv output/netload_only/std-S4-WEST.csv output/netload_only/train-time-S4-WEST output/netload_only/test-time-S4-WEST output/netload_only/tree-S4-WEST.svg &: settings/netload_only/S4-WEST.toml netload_only_train/feature-S4-WEST.csv netload_only_train/target-S4-WEST.csv netload_only_test/feature-S4-WEST.csv
	./run_model.py --config=settings/netload_only/S4-WEST.toml --train-feature=netload_only_train/feature-S4-WEST.csv --train-target=netload_only_train/target-S4-WEST.csv --test-feature=netload_only_test/feature-S4-WEST.csv --predict-mean=output/netload_only/mean-S4-WEST.csv --predict-std=output/netload_only/std-S4-WEST.csv --visualize-tree=output/netload_only/tree-S4-WEST.svg --train-time=output/netload_only/train-time-S4-WEST --test-time=output/netload_only/test-time-S4-WEST 

.PHONY: show-netload_only-S4-WEST
show-netload_only-S4-WEST : output/netload_only/mean-S4-WEST.csv output/netload_only/std-S4-WEST.csv netload_only_test/target-S4-WEST.csv
	./plot_prediction.py --mean=output/netload_only/mean-S4-WEST.csv --std=output/netload_only/std-S4-WEST.csv --target=netload_only_test/target-S4-WEST.csv 

.PHONY: run-netload_only-S2
run-netload_only-S2 : output/netload_only/mean-S2.csv output/netload_only/std-S2.csv output/netload_only/train-time-S2 output/netload_only/test-time-S2 output/netload_only/tree-S2.svg

all : output/netload_only/mean-S2.csv output/netload_only/std-S2.csv output/netload_only/train-time-S2 output/netload_only/test-time-S2 output/netload_only/tree-S2.svg

output/netload_only/mean-S2.csv output/netload_only/std-S2.csv output/netload_only/train-time-S2 output/netload_only/test-time-S2 output/netload_only/tree-S2.svg &: settings/netload_only/S2.toml netload_only_train/feature-S2.csv netload_only_train/target-S2.csv netload_only_test/feature-S2.csv
	./run_model.py --config=settings/netload_only/S2.toml --train-feature=netload_only_train/feature-S2.csv --train-target=netload_only_train/target-S2.csv --test-feature=netload_only_test/feature-S2.csv --predict-mean=output/netload_only/mean-S2.csv --predict-std=output/netload_only/std-S2.csv --visualize-tree=output/netload_only/tree-S2.svg --train-time=output/netload_only/train-time-S2 --test-time=output/netload_only/test-time-S2 

.PHONY: show-netload_only-S2
show-netload_only-S2 : output/netload_only/mean-S2.csv output/netload_only/std-S2.csv netload_only_test/target-S2.csv
	./plot_prediction.py --mean=output/netload_only/mean-S2.csv --std=output/netload_only/std-S2.csv --target=netload_only_test/target-S2.csv 

.PHONY: run-netload_only-S3-CZ
run-netload_only-S3-CZ : output/netload_only/mean-S3-CZ.csv output/netload_only/std-S3-CZ.csv output/netload_only/train-time-S3-CZ output/netload_only/test-time-S3-CZ output/netload_only/tree-S3-CZ.svg

all : output/netload_only/mean-S3-CZ.csv output/netload_only/std-S3-CZ.csv output/netload_only/train-time-S3-CZ output/netload_only/test-time-S3-CZ output/netload_only/tree-S3-CZ.svg

output/netload_only/mean-S3-CZ.csv output/netload_only/std-S3-CZ.csv output/netload_only/train-time-S3-CZ output/netload_only/test-time-S3-CZ output/netload_only/tree-S3-CZ.svg &: settings/netload_only/S3-CZ.toml netload_only_train/feature-S3-CZ.csv netload_only_train/target-S3-CZ.csv netload_only_test/feature-S3-CZ.csv
	./run_model.py --config=settings/netload_only/S3-CZ.toml --train-feature=netload_only_train/feature-S3-CZ.csv --train-target=netload_only_train/target-S3-CZ.csv --test-feature=netload_only_test/feature-S3-CZ.csv --predict-mean=output/netload_only/mean-S3-CZ.csv --predict-std=output/netload_only/std-S3-CZ.csv --visualize-tree=output/netload_only/tree-S3-CZ.svg --train-time=output/netload_only/train-time-S3-CZ --test-time=output/netload_only/test-time-S3-CZ 

.PHONY: show-netload_only-S3-CZ
show-netload_only-S3-CZ : output/netload_only/mean-S3-CZ.csv output/netload_only/std-S3-CZ.csv netload_only_test/target-S3-CZ.csv
	./plot_prediction.py --mean=output/netload_only/mean-S3-CZ.csv --std=output/netload_only/std-S3-CZ.csv --target=netload_only_test/target-S3-CZ.csv 

.PHONY: run-netload_only-S4-SOUTH
run-netload_only-S4-SOUTH : output/netload_only/mean-S4-SOUTH.csv output/netload_only/std-S4-SOUTH.csv output/netload_only/train-time-S4-SOUTH output/netload_only/test-time-S4-SOUTH output/netload_only/tree-S4-SOUTH.svg

all : output/netload_only/mean-S4-SOUTH.csv output/netload_only/std-S4-SOUTH.csv output/netload_only/train-time-S4-SOUTH output/netload_only/test-time-S4-SOUTH output/netload_only/tree-S4-SOUTH.svg

output/netload_only/mean-S4-SOUTH.csv output/netload_only/std-S4-SOUTH.csv output/netload_only/train-time-S4-SOUTH output/netload_only/test-time-S4-SOUTH output/netload_only/tree-S4-SOUTH.svg &: settings/netload_only/S4-SOUTH.toml netload_only_train/feature-S4-SOUTH.csv netload_only_train/target-S4-SOUTH.csv netload_only_test/feature-S4-SOUTH.csv
	./run_model.py --config=settings/netload_only/S4-SOUTH.toml --train-feature=netload_only_train/feature-S4-SOUTH.csv --train-target=netload_only_train/target-S4-SOUTH.csv --test-feature=netload_only_test/feature-S4-SOUTH.csv --predict-mean=output/netload_only/mean-S4-SOUTH.csv --predict-std=output/netload_only/std-S4-SOUTH.csv --visualize-tree=output/netload_only/tree-S4-SOUTH.svg --train-time=output/netload_only/train-time-S4-SOUTH --test-time=output/netload_only/test-time-S4-SOUTH 

.PHONY: show-netload_only-S4-SOUTH
show-netload_only-S4-SOUTH : output/netload_only/mean-S4-SOUTH.csv output/netload_only/std-S4-SOUTH.csv netload_only_test/target-S4-SOUTH.csv
	./plot_prediction.py --mean=output/netload_only/mean-S4-SOUTH.csv --std=output/netload_only/std-S4-SOUTH.csv --target=netload_only_test/target-S4-SOUTH.csv 

.PHONY: run-combined-S3-BG
run-combined-S3-BG : output/combined/mean-S3-BG.csv output/combined/std-S3-BG.csv output/combined/train-time-S3-BG output/combined/test-time-S3-BG output/combined/tree-S3-BG.svg

all : output/combined/mean-S3-BG.csv output/combined/std-S3-BG.csv output/combined/train-time-S3-BG output/combined/test-time-S3-BG output/combined/tree-S3-BG.svg

output/combined/mean-S3-BG.csv output/combined/std-S3-BG.csv output/combined/train-time-S3-BG output/combined/test-time-S3-BG output/combined/tree-S3-BG.svg &: settings/combined/S3-BG.toml combined_train/feature-S3-BG.csv combined_train/target-S3-BG.csv combined_test/feature-S3-BG.csv
	./run_model.py --config=settings/combined/S3-BG.toml --train-feature=combined_train/feature-S3-BG.csv --train-target=combined_train/target-S3-BG.csv --test-feature=combined_test/feature-S3-BG.csv --predict-mean=output/combined/mean-S3-BG.csv --predict-std=output/combined/std-S3-BG.csv --visualize-tree=output/combined/tree-S3-BG.svg --train-time=output/combined/train-time-S3-BG --test-time=output/combined/test-time-S3-BG 

.PHONY: show-combined-S3-BG
show-combined-S3-BG : output/combined/mean-S3-BG.csv output/combined/std-S3-BG.csv combined_test/target-S3-BG.csv
	./plot_prediction.py --mean=output/combined/mean-S3-BG.csv --std=output/combined/std-S3-BG.csv --target=combined_test/target-S3-BG.csv 

.PHONY: run-combined-S3-NL
run-combined-S3-NL : output/combined/mean-S3-NL.csv output/combined/std-S3-NL.csv output/combined/train-time-S3-NL output/combined/test-time-S3-NL output/combined/tree-S3-NL.svg

all : output/combined/mean-S3-NL.csv output/combined/std-S3-NL.csv output/combined/train-time-S3-NL output/combined/test-time-S3-NL output/combined/tree-S3-NL.svg

output/combined/mean-S3-NL.csv output/combined/std-S3-NL.csv output/combined/train-time-S3-NL output/combined/test-time-S3-NL output/combined/tree-S3-NL.svg &: settings/combined/S3-NL.toml combined_train/feature-S3-NL.csv combined_train/target-S3-NL.csv combined_test/feature-S3-NL.csv
	./run_model.py --config=settings/combined/S3-NL.toml --train-feature=combined_train/feature-S3-NL.csv --train-target=combined_train/target-S3-NL.csv --test-feature=combined_test/feature-S3-NL.csv --predict-mean=output/combined/mean-S3-NL.csv --predict-std=output/combined/std-S3-NL.csv --visualize-tree=output/combined/tree-S3-NL.svg --train-time=output/combined/train-time-S3-NL --test-time=output/combined/test-time-S3-NL 

.PHONY: show-combined-S3-NL
show-combined-S3-NL : output/combined/mean-S3-NL.csv output/combined/std-S3-NL.csv combined_test/target-S3-NL.csv
	./plot_prediction.py --mean=output/combined/mean-S3-NL.csv --std=output/combined/std-S3-NL.csv --target=combined_test/target-S3-NL.csv 

.PHONY: run-combined-S3-AT
run-combined-S3-AT : output/combined/mean-S3-AT.csv output/combined/std-S3-AT.csv output/combined/train-time-S3-AT output/combined/test-time-S3-AT output/combined/tree-S3-AT.svg

all : output/combined/mean-S3-AT.csv output/combined/std-S3-AT.csv output/combined/train-time-S3-AT output/combined/test-time-S3-AT output/combined/tree-S3-AT.svg

output/combined/mean-S3-AT.csv output/combined/std-S3-AT.csv output/combined/train-time-S3-AT output/combined/test-time-S3-AT output/combined/tree-S3-AT.svg &: settings/combined/S3-AT.toml combined_train/feature-S3-AT.csv combined_train/target-S3-AT.csv combined_test/feature-S3-AT.csv
	./run_model.py --config=settings/combined/S3-AT.toml --train-feature=combined_train/feature-S3-AT.csv --train-target=combined_train/target-S3-AT.csv --test-feature=combined_test/feature-S3-AT.csv --predict-mean=output/combined/mean-S3-AT.csv --predict-std=output/combined/std-S3-AT.csv --visualize-tree=output/combined/tree-S3-AT.svg --train-time=output/combined/train-time-S3-AT --test-time=output/combined/test-time-S3-AT 

.PHONY: show-combined-S3-AT
show-combined-S3-AT : output/combined/mean-S3-AT.csv output/combined/std-S3-AT.csv combined_test/target-S3-AT.csv
	./plot_prediction.py --mean=output/combined/mean-S3-AT.csv --std=output/combined/std-S3-AT.csv --target=combined_test/target-S3-AT.csv 

.PHONY: run-combined-S3-ES
run-combined-S3-ES : output/combined/mean-S3-ES.csv output/combined/std-S3-ES.csv output/combined/train-time-S3-ES output/combined/test-time-S3-ES output/combined/tree-S3-ES.svg

all : output/combined/mean-S3-ES.csv output/combined/std-S3-ES.csv output/combined/train-time-S3-ES output/combined/test-time-S3-ES output/combined/tree-S3-ES.svg

output/combined/mean-S3-ES.csv output/combined/std-S3-ES.csv output/combined/train-time-S3-ES output/combined/test-time-S3-ES output/combined/tree-S3-ES.svg &: settings/combined/S3-ES.toml combined_train/feature-S3-ES.csv combined_train/target-S3-ES.csv combined_test/feature-S3-ES.csv
	./run_model.py --config=settings/combined/S3-ES.toml --train-feature=combined_train/feature-S3-ES.csv --train-target=combined_train/target-S3-ES.csv --test-feature=combined_test/feature-S3-ES.csv --predict-mean=output/combined/mean-S3-ES.csv --predict-std=output/combined/std-S3-ES.csv --visualize-tree=output/combined/tree-S3-ES.svg --train-time=output/combined/train-time-S3-ES --test-time=output/combined/test-time-S3-ES 

.PHONY: show-combined-S3-ES
show-combined-S3-ES : output/combined/mean-S3-ES.csv output/combined/std-S3-ES.csv combined_test/target-S3-ES.csv
	./plot_prediction.py --mean=output/combined/mean-S3-ES.csv --std=output/combined/std-S3-ES.csv --target=combined_test/target-S3-ES.csv 

.PHONY: run-combined-S3-GR
run-combined-S3-GR : output/combined/mean-S3-GR.csv output/combined/std-S3-GR.csv output/combined/train-time-S3-GR output/combined/test-time-S3-GR output/combined/tree-S3-GR.svg

all : output/combined/mean-S3-GR.csv output/combined/std-S3-GR.csv output/combined/train-time-S3-GR output/combined/test-time-S3-GR output/combined/tree-S3-GR.svg

output/combined/mean-S3-GR.csv output/combined/std-S3-GR.csv output/combined/train-time-S3-GR output/combined/test-time-S3-GR output/combined/tree-S3-GR.svg &: settings/combined/S3-GR.toml combined_train/feature-S3-GR.csv combined_train/target-S3-GR.csv combined_test/feature-S3-GR.csv
	./run_model.py --config=settings/combined/S3-GR.toml --train-feature=combined_train/feature-S3-GR.csv --train-target=combined_train/target-S3-GR.csv --test-feature=combined_test/feature-S3-GR.csv --predict-mean=output/combined/mean-S3-GR.csv --predict-std=output/combined/std-S3-GR.csv --visualize-tree=output/combined/tree-S3-GR.svg --train-time=output/combined/train-time-S3-GR --test-time=output/combined/test-time-S3-GR 

.PHONY: show-combined-S3-GR
show-combined-S3-GR : output/combined/mean-S3-GR.csv output/combined/std-S3-GR.csv combined_test/target-S3-GR.csv
	./plot_prediction.py --mean=output/combined/mean-S3-GR.csv --std=output/combined/std-S3-GR.csv --target=combined_test/target-S3-GR.csv 

.PHONY: run-combined-S3-IT
run-combined-S3-IT : output/combined/mean-S3-IT.csv output/combined/std-S3-IT.csv output/combined/train-time-S3-IT output/combined/test-time-S3-IT output/combined/tree-S3-IT.svg

all : output/combined/mean-S3-IT.csv output/combined/std-S3-IT.csv output/combined/train-time-S3-IT output/combined/test-time-S3-IT output/combined/tree-S3-IT.svg

output/combined/mean-S3-IT.csv output/combined/std-S3-IT.csv output/combined/train-time-S3-IT output/combined/test-time-S3-IT output/combined/tree-S3-IT.svg &: settings/combined/S3-IT.toml combined_train/feature-S3-IT.csv combined_train/target-S3-IT.csv combined_test/feature-S3-IT.csv
	./run_model.py --config=settings/combined/S3-IT.toml --train-feature=combined_train/feature-S3-IT.csv --train-target=combined_train/target-S3-IT.csv --test-feature=combined_test/feature-S3-IT.csv --predict-mean=output/combined/mean-S3-IT.csv --predict-std=output/combined/std-S3-IT.csv --visualize-tree=output/combined/tree-S3-IT.svg --train-time=output/combined/train-time-S3-IT --test-time=output/combined/test-time-S3-IT 

.PHONY: show-combined-S3-IT
show-combined-S3-IT : output/combined/mean-S3-IT.csv output/combined/std-S3-IT.csv combined_test/target-S3-IT.csv
	./plot_prediction.py --mean=output/combined/mean-S3-IT.csv --std=output/combined/std-S3-IT.csv --target=combined_test/target-S3-IT.csv 

.PHONY: run-combined-S3-SI
run-combined-S3-SI : output/combined/mean-S3-SI.csv output/combined/std-S3-SI.csv output/combined/train-time-S3-SI output/combined/test-time-S3-SI output/combined/tree-S3-SI.svg

all : output/combined/mean-S3-SI.csv output/combined/std-S3-SI.csv output/combined/train-time-S3-SI output/combined/test-time-S3-SI output/combined/tree-S3-SI.svg

output/combined/mean-S3-SI.csv output/combined/std-S3-SI.csv output/combined/train-time-S3-SI output/combined/test-time-S3-SI output/combined/tree-S3-SI.svg &: settings/combined/S3-SI.toml combined_train/feature-S3-SI.csv combined_train/target-S3-SI.csv combined_test/feature-S3-SI.csv
	./run_model.py --config=settings/combined/S3-SI.toml --train-feature=combined_train/feature-S3-SI.csv --train-target=combined_train/target-S3-SI.csv --test-feature=combined_test/feature-S3-SI.csv --predict-mean=output/combined/mean-S3-SI.csv --predict-std=output/combined/std-S3-SI.csv --visualize-tree=output/combined/tree-S3-SI.svg --train-time=output/combined/train-time-S3-SI --test-time=output/combined/test-time-S3-SI 

.PHONY: show-combined-S3-SI
show-combined-S3-SI : output/combined/mean-S3-SI.csv output/combined/std-S3-SI.csv combined_test/target-S3-SI.csv
	./plot_prediction.py --mean=output/combined/mean-S3-SI.csv --std=output/combined/std-S3-SI.csv --target=combined_test/target-S3-SI.csv 

.PHONY: run-combined-S4-MIDATL
run-combined-S4-MIDATL : output/combined/mean-S4-MIDATL.csv output/combined/std-S4-MIDATL.csv output/combined/train-time-S4-MIDATL output/combined/test-time-S4-MIDATL output/combined/tree-S4-MIDATL.svg

all : output/combined/mean-S4-MIDATL.csv output/combined/std-S4-MIDATL.csv output/combined/train-time-S4-MIDATL output/combined/test-time-S4-MIDATL output/combined/tree-S4-MIDATL.svg

output/combined/mean-S4-MIDATL.csv output/combined/std-S4-MIDATL.csv output/combined/train-time-S4-MIDATL output/combined/test-time-S4-MIDATL output/combined/tree-S4-MIDATL.svg &: settings/combined/S4-MIDATL.toml combined_train/feature-S4-MIDATL.csv combined_train/target-S4-MIDATL.csv combined_test/feature-S4-MIDATL.csv
	./run_model.py --config=settings/combined/S4-MIDATL.toml --train-feature=combined_train/feature-S4-MIDATL.csv --train-target=combined_train/target-S4-MIDATL.csv --test-feature=combined_test/feature-S4-MIDATL.csv --predict-mean=output/combined/mean-S4-MIDATL.csv --predict-std=output/combined/std-S4-MIDATL.csv --visualize-tree=output/combined/tree-S4-MIDATL.svg --train-time=output/combined/train-time-S4-MIDATL --test-time=output/combined/test-time-S4-MIDATL 

.PHONY: show-combined-S4-MIDATL
show-combined-S4-MIDATL : output/combined/mean-S4-MIDATL.csv output/combined/std-S4-MIDATL.csv combined_test/target-S4-MIDATL.csv
	./plot_prediction.py --mean=output/combined/mean-S4-MIDATL.csv --std=output/combined/std-S4-MIDATL.csv --target=combined_test/target-S4-MIDATL.csv 

.PHONY: run-combined-S3-PT
run-combined-S3-PT : output/combined/mean-S3-PT.csv output/combined/std-S3-PT.csv output/combined/train-time-S3-PT output/combined/test-time-S3-PT output/combined/tree-S3-PT.svg

all : output/combined/mean-S3-PT.csv output/combined/std-S3-PT.csv output/combined/train-time-S3-PT output/combined/test-time-S3-PT output/combined/tree-S3-PT.svg

output/combined/mean-S3-PT.csv output/combined/std-S3-PT.csv output/combined/train-time-S3-PT output/combined/test-time-S3-PT output/combined/tree-S3-PT.svg &: settings/combined/S3-PT.toml combined_train/feature-S3-PT.csv combined_train/target-S3-PT.csv combined_test/feature-S3-PT.csv
	./run_model.py --config=settings/combined/S3-PT.toml --train-feature=combined_train/feature-S3-PT.csv --train-target=combined_train/target-S3-PT.csv --test-feature=combined_test/feature-S3-PT.csv --predict-mean=output/combined/mean-S3-PT.csv --predict-std=output/combined/std-S3-PT.csv --visualize-tree=output/combined/tree-S3-PT.svg --train-time=output/combined/train-time-S3-PT --test-time=output/combined/test-time-S3-PT 

.PHONY: show-combined-S3-PT
show-combined-S3-PT : output/combined/mean-S3-PT.csv output/combined/std-S3-PT.csv combined_test/target-S3-PT.csv
	./plot_prediction.py --mean=output/combined/mean-S3-PT.csv --std=output/combined/std-S3-PT.csv --target=combined_test/target-S3-PT.csv 

.PHONY: run-combined-S1
run-combined-S1 : output/combined/mean-S1.csv output/combined/std-S1.csv output/combined/train-time-S1 output/combined/test-time-S1 output/combined/tree-S1.svg

all : output/combined/mean-S1.csv output/combined/std-S1.csv output/combined/train-time-S1 output/combined/test-time-S1 output/combined/tree-S1.svg

output/combined/mean-S1.csv output/combined/std-S1.csv output/combined/train-time-S1 output/combined/test-time-S1 output/combined/tree-S1.svg &: settings/combined/S1.toml combined_train/feature-S1.csv combined_train/target-S1.csv combined_test/feature-S1.csv
	./run_model.py --config=settings/combined/S1.toml --train-feature=combined_train/feature-S1.csv --train-target=combined_train/target-S1.csv --test-feature=combined_test/feature-S1.csv --predict-mean=output/combined/mean-S1.csv --predict-std=output/combined/std-S1.csv --visualize-tree=output/combined/tree-S1.svg --train-time=output/combined/train-time-S1 --test-time=output/combined/test-time-S1 

.PHONY: show-combined-S1
show-combined-S1 : output/combined/mean-S1.csv output/combined/std-S1.csv combined_test/target-S1.csv
	./plot_prediction.py --mean=output/combined/mean-S1.csv --std=output/combined/std-S1.csv --target=combined_test/target-S1.csv 

.PHONY: run-combined-S3-CH
run-combined-S3-CH : output/combined/mean-S3-CH.csv output/combined/std-S3-CH.csv output/combined/train-time-S3-CH output/combined/test-time-S3-CH output/combined/tree-S3-CH.svg

all : output/combined/mean-S3-CH.csv output/combined/std-S3-CH.csv output/combined/train-time-S3-CH output/combined/test-time-S3-CH output/combined/tree-S3-CH.svg

output/combined/mean-S3-CH.csv output/combined/std-S3-CH.csv output/combined/train-time-S3-CH output/combined/test-time-S3-CH output/combined/tree-S3-CH.svg &: settings/combined/S3-CH.toml combined_train/feature-S3-CH.csv combined_train/target-S3-CH.csv combined_test/feature-S3-CH.csv
	./run_model.py --config=settings/combined/S3-CH.toml --train-feature=combined_train/feature-S3-CH.csv --train-target=combined_train/target-S3-CH.csv --test-feature=combined_test/feature-S3-CH.csv --predict-mean=output/combined/mean-S3-CH.csv --predict-std=output/combined/std-S3-CH.csv --visualize-tree=output/combined/tree-S3-CH.svg --train-time=output/combined/train-time-S3-CH --test-time=output/combined/test-time-S3-CH 

.PHONY: show-combined-S3-CH
show-combined-S3-CH : output/combined/mean-S3-CH.csv output/combined/std-S3-CH.csv combined_test/target-S3-CH.csv
	./plot_prediction.py --mean=output/combined/mean-S3-CH.csv --std=output/combined/std-S3-CH.csv --target=combined_test/target-S3-CH.csv 

.PHONY: run-combined-S3-SK
run-combined-S3-SK : output/combined/mean-S3-SK.csv output/combined/std-S3-SK.csv output/combined/train-time-S3-SK output/combined/test-time-S3-SK output/combined/tree-S3-SK.svg

all : output/combined/mean-S3-SK.csv output/combined/std-S3-SK.csv output/combined/train-time-S3-SK output/combined/test-time-S3-SK output/combined/tree-S3-SK.svg

output/combined/mean-S3-SK.csv output/combined/std-S3-SK.csv output/combined/train-time-S3-SK output/combined/test-time-S3-SK output/combined/tree-S3-SK.svg &: settings/combined/S3-SK.toml combined_train/feature-S3-SK.csv combined_train/target-S3-SK.csv combined_test/feature-S3-SK.csv
	./run_model.py --config=settings/combined/S3-SK.toml --train-feature=combined_train/feature-S3-SK.csv --train-target=combined_train/target-S3-SK.csv --test-feature=combined_test/feature-S3-SK.csv --predict-mean=output/combined/mean-S3-SK.csv --predict-std=output/combined/std-S3-SK.csv --visualize-tree=output/combined/tree-S3-SK.svg --train-time=output/combined/train-time-S3-SK --test-time=output/combined/test-time-S3-SK 

.PHONY: show-combined-S3-SK
show-combined-S3-SK : output/combined/mean-S3-SK.csv output/combined/std-S3-SK.csv combined_test/target-S3-SK.csv
	./plot_prediction.py --mean=output/combined/mean-S3-SK.csv --std=output/combined/std-S3-SK.csv --target=combined_test/target-S3-SK.csv 

.PHONY: run-combined-S3-DK
run-combined-S3-DK : output/combined/mean-S3-DK.csv output/combined/std-S3-DK.csv output/combined/train-time-S3-DK output/combined/test-time-S3-DK output/combined/tree-S3-DK.svg

all : output/combined/mean-S3-DK.csv output/combined/std-S3-DK.csv output/combined/train-time-S3-DK output/combined/test-time-S3-DK output/combined/tree-S3-DK.svg

output/combined/mean-S3-DK.csv output/combined/std-S3-DK.csv output/combined/train-time-S3-DK output/combined/test-time-S3-DK output/combined/tree-S3-DK.svg &: settings/combined/S3-DK.toml combined_train/feature-S3-DK.csv combined_train/target-S3-DK.csv combined_test/feature-S3-DK.csv
	./run_model.py --config=settings/combined/S3-DK.toml --train-feature=combined_train/feature-S3-DK.csv --train-target=combined_train/target-S3-DK.csv --test-feature=combined_test/feature-S3-DK.csv --predict-mean=output/combined/mean-S3-DK.csv --predict-std=output/combined/std-S3-DK.csv --visualize-tree=output/combined/tree-S3-DK.svg --train-time=output/combined/train-time-S3-DK --test-time=output/combined/test-time-S3-DK 

.PHONY: show-combined-S3-DK
show-combined-S3-DK : output/combined/mean-S3-DK.csv output/combined/std-S3-DK.csv combined_test/target-S3-DK.csv
	./plot_prediction.py --mean=output/combined/mean-S3-DK.csv --std=output/combined/std-S3-DK.csv --target=combined_test/target-S3-DK.csv 

.PHONY: run-combined-S3-FR
run-combined-S3-FR : output/combined/mean-S3-FR.csv output/combined/std-S3-FR.csv output/combined/train-time-S3-FR output/combined/test-time-S3-FR output/combined/tree-S3-FR.svg

all : output/combined/mean-S3-FR.csv output/combined/std-S3-FR.csv output/combined/train-time-S3-FR output/combined/test-time-S3-FR output/combined/tree-S3-FR.svg

output/combined/mean-S3-FR.csv output/combined/std-S3-FR.csv output/combined/train-time-S3-FR output/combined/test-time-S3-FR output/combined/tree-S3-FR.svg &: settings/combined/S3-FR.toml combined_train/feature-S3-FR.csv combined_train/target-S3-FR.csv combined_test/feature-S3-FR.csv
	./run_model.py --config=settings/combined/S3-FR.toml --train-feature=combined_train/feature-S3-FR.csv --train-target=combined_train/target-S3-FR.csv --test-feature=combined_test/feature-S3-FR.csv --predict-mean=output/combined/mean-S3-FR.csv --predict-std=output/combined/std-S3-FR.csv --visualize-tree=output/combined/tree-S3-FR.svg --train-time=output/combined/train-time-S3-FR --test-time=output/combined/test-time-S3-FR 

.PHONY: show-combined-S3-FR
show-combined-S3-FR : output/combined/mean-S3-FR.csv output/combined/std-S3-FR.csv combined_test/target-S3-FR.csv
	./plot_prediction.py --mean=output/combined/mean-S3-FR.csv --std=output/combined/std-S3-FR.csv --target=combined_test/target-S3-FR.csv 

.PHONY: run-combined-S3-BE
run-combined-S3-BE : output/combined/mean-S3-BE.csv output/combined/std-S3-BE.csv output/combined/train-time-S3-BE output/combined/test-time-S3-BE output/combined/tree-S3-BE.svg

all : output/combined/mean-S3-BE.csv output/combined/std-S3-BE.csv output/combined/train-time-S3-BE output/combined/test-time-S3-BE output/combined/tree-S3-BE.svg

output/combined/mean-S3-BE.csv output/combined/std-S3-BE.csv output/combined/train-time-S3-BE output/combined/test-time-S3-BE output/combined/tree-S3-BE.svg &: settings/combined/S3-BE.toml combined_train/feature-S3-BE.csv combined_train/target-S3-BE.csv combined_test/feature-S3-BE.csv
	./run_model.py --config=settings/combined/S3-BE.toml --train-feature=combined_train/feature-S3-BE.csv --train-target=combined_train/target-S3-BE.csv --test-feature=combined_test/feature-S3-BE.csv --predict-mean=output/combined/mean-S3-BE.csv --predict-std=output/combined/std-S3-BE.csv --visualize-tree=output/combined/tree-S3-BE.svg --train-time=output/combined/train-time-S3-BE --test-time=output/combined/test-time-S3-BE 

.PHONY: show-combined-S3-BE
show-combined-S3-BE : output/combined/mean-S3-BE.csv output/combined/std-S3-BE.csv combined_test/target-S3-BE.csv
	./plot_prediction.py --mean=output/combined/mean-S3-BE.csv --std=output/combined/std-S3-BE.csv --target=combined_test/target-S3-BE.csv 

.PHONY: run-combined-S4-WEST
run-combined-S4-WEST : output/combined/mean-S4-WEST.csv output/combined/std-S4-WEST.csv output/combined/train-time-S4-WEST output/combined/test-time-S4-WEST output/combined/tree-S4-WEST.svg

all : output/combined/mean-S4-WEST.csv output/combined/std-S4-WEST.csv output/combined/train-time-S4-WEST output/combined/test-time-S4-WEST output/combined/tree-S4-WEST.svg

output/combined/mean-S4-WEST.csv output/combined/std-S4-WEST.csv output/combined/train-time-S4-WEST output/combined/test-time-S4-WEST output/combined/tree-S4-WEST.svg &: settings/combined/S4-WEST.toml combined_train/feature-S4-WEST.csv combined_train/target-S4-WEST.csv combined_test/feature-S4-WEST.csv
	./run_model.py --config=settings/combined/S4-WEST.toml --train-feature=combined_train/feature-S4-WEST.csv --train-target=combined_train/target-S4-WEST.csv --test-feature=combined_test/feature-S4-WEST.csv --predict-mean=output/combined/mean-S4-WEST.csv --predict-std=output/combined/std-S4-WEST.csv --visualize-tree=output/combined/tree-S4-WEST.svg --train-time=output/combined/train-time-S4-WEST --test-time=output/combined/test-time-S4-WEST 

.PHONY: show-combined-S4-WEST
show-combined-S4-WEST : output/combined/mean-S4-WEST.csv output/combined/std-S4-WEST.csv combined_test/target-S4-WEST.csv
	./plot_prediction.py --mean=output/combined/mean-S4-WEST.csv --std=output/combined/std-S4-WEST.csv --target=combined_test/target-S4-WEST.csv 

.PHONY: run-combined-S2
run-combined-S2 : output/combined/mean-S2.csv output/combined/std-S2.csv output/combined/train-time-S2 output/combined/test-time-S2 output/combined/tree-S2.svg

all : output/combined/mean-S2.csv output/combined/std-S2.csv output/combined/train-time-S2 output/combined/test-time-S2 output/combined/tree-S2.svg

output/combined/mean-S2.csv output/combined/std-S2.csv output/combined/train-time-S2 output/combined/test-time-S2 output/combined/tree-S2.svg &: settings/combined/S2.toml combined_train/feature-S2.csv combined_train/target-S2.csv combined_test/feature-S2.csv
	./run_model.py --config=settings/combined/S2.toml --train-feature=combined_train/feature-S2.csv --train-target=combined_train/target-S2.csv --test-feature=combined_test/feature-S2.csv --predict-mean=output/combined/mean-S2.csv --predict-std=output/combined/std-S2.csv --visualize-tree=output/combined/tree-S2.svg --train-time=output/combined/train-time-S2 --test-time=output/combined/test-time-S2 

.PHONY: show-combined-S2
show-combined-S2 : output/combined/mean-S2.csv output/combined/std-S2.csv combined_test/target-S2.csv
	./plot_prediction.py --mean=output/combined/mean-S2.csv --std=output/combined/std-S2.csv --target=combined_test/target-S2.csv 

.PHONY: run-combined-S3-CZ
run-combined-S3-CZ : output/combined/mean-S3-CZ.csv output/combined/std-S3-CZ.csv output/combined/train-time-S3-CZ output/combined/test-time-S3-CZ output/combined/tree-S3-CZ.svg

all : output/combined/mean-S3-CZ.csv output/combined/std-S3-CZ.csv output/combined/train-time-S3-CZ output/combined/test-time-S3-CZ output/combined/tree-S3-CZ.svg

output/combined/mean-S3-CZ.csv output/combined/std-S3-CZ.csv output/combined/train-time-S3-CZ output/combined/test-time-S3-CZ output/combined/tree-S3-CZ.svg &: settings/combined/S3-CZ.toml combined_train/feature-S3-CZ.csv combined_train/target-S3-CZ.csv combined_test/feature-S3-CZ.csv
	./run_model.py --config=settings/combined/S3-CZ.toml --train-feature=combined_train/feature-S3-CZ.csv --train-target=combined_train/target-S3-CZ.csv --test-feature=combined_test/feature-S3-CZ.csv --predict-mean=output/combined/mean-S3-CZ.csv --predict-std=output/combined/std-S3-CZ.csv --visualize-tree=output/combined/tree-S3-CZ.svg --train-time=output/combined/train-time-S3-CZ --test-time=output/combined/test-time-S3-CZ 

.PHONY: show-combined-S3-CZ
show-combined-S3-CZ : output/combined/mean-S3-CZ.csv output/combined/std-S3-CZ.csv combined_test/target-S3-CZ.csv
	./plot_prediction.py --mean=output/combined/mean-S3-CZ.csv --std=output/combined/std-S3-CZ.csv --target=combined_test/target-S3-CZ.csv 

.PHONY: run-combined-S4-SOUTH
run-combined-S4-SOUTH : output/combined/mean-S4-SOUTH.csv output/combined/std-S4-SOUTH.csv output/combined/train-time-S4-SOUTH output/combined/test-time-S4-SOUTH output/combined/tree-S4-SOUTH.svg

all : output/combined/mean-S4-SOUTH.csv output/combined/std-S4-SOUTH.csv output/combined/train-time-S4-SOUTH output/combined/test-time-S4-SOUTH output/combined/tree-S4-SOUTH.svg

output/combined/mean-S4-SOUTH.csv output/combined/std-S4-SOUTH.csv output/combined/train-time-S4-SOUTH output/combined/test-time-S4-SOUTH output/combined/tree-S4-SOUTH.svg &: settings/combined/S4-SOUTH.toml combined_train/feature-S4-SOUTH.csv combined_train/target-S4-SOUTH.csv combined_test/feature-S4-SOUTH.csv
	./run_model.py --config=settings/combined/S4-SOUTH.toml --train-feature=combined_train/feature-S4-SOUTH.csv --train-target=combined_train/target-S4-SOUTH.csv --test-feature=combined_test/feature-S4-SOUTH.csv --predict-mean=output/combined/mean-S4-SOUTH.csv --predict-std=output/combined/std-S4-SOUTH.csv --visualize-tree=output/combined/tree-S4-SOUTH.svg --train-time=output/combined/train-time-S4-SOUTH --test-time=output/combined/test-time-S4-SOUTH 

.PHONY: show-combined-S4-SOUTH
show-combined-S4-SOUTH : output/combined/mean-S4-SOUTH.csv output/combined/std-S4-SOUTH.csv combined_test/target-S4-SOUTH.csv
	./plot_prediction.py --mean=output/combined/mean-S4-SOUTH.csv --std=output/combined/std-S4-SOUTH.csv --target=combined_test/target-S4-SOUTH.csv 

