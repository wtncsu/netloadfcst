.DEFAULT_GOAL:=all

.PHONY: all
all: all-dt all-fdt all-bdt all-bfdt

.PHONY: clean
clean:
	rm -rf output

#-------------------------------------------------------------------------------
# PREPROCESS
#-------------------------------------------------------------------------------
pp:=$(patsubst config-pp/%.ini,%,$(wildcard config-pp/*.ini))
define gen_pp
.PHONY: pp-$(1)
pp-$(1): output/pp-$(1).done

output/pp-$(1).done: config-pp/$(1).ini dataset/$(1).csv
	./preprocess.py $$< && touch $$@
endef
$(foreach t,$(pp),$(eval $(call gen_pp,$(t))))

.PHONY: all-pp
all-pp: $(patsubst %,pp-%,$(pp))

#-------------------------------------------------------------------------------
# DECISION TREE
#-------------------------------------------------------------------------------
dt:=$(patsubst config-dt/%.ini,%,$(wildcard config-dt/*.ini))

define gen_dt
dt_timing_files+=output/dt-timing-$(1).csv
dt_train_metrics_files+=output/dt-train-metrics-$(1).csv
dt_test_metrics_files+=output/dt-test-metrics-$(1).csv

.PHONY: dt-$(1)
dt-$(1): output/dt-$(1).done

output/dt-$(1).done: config-dt/$(1).ini output/pp-$(1).done
	./fit_dt.py $$< && touch $$@

.PHONY: plot-dt-train-$(1)
plot-dt-train-$(1): output/dt-$(1).done
	./plot_point_pred.py --pred output/dt-train-pred-$(1).csv \
	--target output/train-target-$(1).csv

.PHONY: plot-dt-test-$(1)
plot-dt-test-$(1): output/dt-$(1).done
	./plot_point_pred.py --pred output/dt-test-pred-$(1).csv \
	--target output/test-target-$(1).csv

output/dt-timing-$(1).csv: output/dt-$(1).done

output/dt-train-metrics-$(1).csv: output/dt-$(1).done
	./calc_point_metrics.py --out $$@ --pred output/dt-train-pred-$(1).csv \
	--target output/train-target-$(1).csv

output/dt-test-metrics-$(1).csv: output/dt-$(1).done
	./calc_point_metrics.py --out $$@ --pred output/dt-test-pred-$(1).csv \
	--target output/test-target-$(1).csv
endef
$(foreach t,$(dt),$(eval $(call gen_dt,$(t))))

.PHONY: all-dt
all-dt: $(patsubst %,dt-%,$(dt)) \
all-dt-timing all-dt-train-metrics all-dt-test-metrics

.PHONY: all-dt-timing
all-dt-timing: output/dt-timing.csv

.PHONY: all-dt-train-metrics
all-dt-train-metrics: output/dt-train-metrics.csv

.PHONY: all-dt-test-metrics
all-dt-test-metrics: output/dt-test-metrics.csv

output/dt-timing.csv: $(dt_timing_files)
	./gen_report.py --out $@ $^

output/dt-train-metrics.csv: $(dt_train_metrics_files)
	./gen_report.py --out $@ $^

output/dt-test-metrics.csv: $(dt_test_metrics_files)
	./gen_report.py --out $@ $^

#-------------------------------------------------------------------------------
# FUZZY DECISION TREE
#-------------------------------------------------------------------------------
fdt:=$(patsubst config-fdt/%.ini,%,$(wildcard config-fdt/*.ini))

define gen_fdt
fdt_timing_files+=output/fdt-timing-$(1).csv
fdt_train_metrics_files+=output/fdt-train-metrics-$(1).csv
fdt_test_metrics_files+=output/fdt-test-metrics-$(1).csv

.PHONY: fdt-$(1)
fdt-$(1): output/fdt-$(1).done

output/fdt-$(1).done: config-fdt/$(1).ini output/pp-$(1).done
	./fit_fdt.py $$< && touch $$@

.PHONY: plot-fdt-train-$(1)
plot-fdt-train-$(1): output/fdt-$(1).done
	./plot_point_pred.py --pred output/fdt-train-pred-$(1).csv \
	--target output/train-target-$(1).csv

.PHONY: plot-fdt-test-$(1)
plot-fdt-test-$(1): output/fdt-$(1).done
	./plot_point_pred.py --pred output/fdt-test-pred-$(1).csv \
	--target output/test-target-$(1).csv

output/fdt-timing-$(1).csv: output/fdt-$(1).done

output/fdt-train-metrics-$(1).csv: output/fdt-$(1).done
	./calc_point_metrics.py --out $$@ --pred output/fdt-train-pred-$(1).csv \
	--target output/train-target-$(1).csv

output/fdt-test-metrics-$(1).csv: output/fdt-$(1).done
	./calc_point_metrics.py --out $$@ --pred output/fdt-test-pred-$(1).csv \
	--target output/test-target-$(1).csv
endef
$(foreach t,$(fdt),$(eval $(call gen_fdt,$(t))))

.PHONY: all-fdt
all-fdt: $(patsubst %,fdt-%,$(fdt)) \
all-fdt-timing all-fdt-train-metrics all-fdt-test-metrics

.PHONY: all-fdt-timing
all-fdt-timing: output/fdt-timing.csv

.PHONY: all-fdt-train-metrics
all-fdt-train-metrics: output/fdt-train-metrics.csv

.PHONY: all-fdt-test-metrics
all-fdt-test-metrics: output/fdt-test-metrics.csv

output/fdt-timing.csv: $(fdt_timing_files)
	./gen_report.py --out $@ $^

output/fdt-train-metrics.csv: $(fdt_train_metrics_files)
	./gen_report.py --out $@ $^

output/fdt-test-metrics.csv: $(fdt_test_metrics_files)
	./gen_report.py --out $@ $^


#-------------------------------------------------------------------------------
# BOOST DECISION TREE
#-------------------------------------------------------------------------------
bdt:=$(patsubst config-bdt/%.ini,%,$(wildcard config-bdt/*.ini))

define gen_bdt
bdt_timing_files+=output/bdt-timing-$(1).csv
bdt_train_metrics_files+=output/bdt-train-metrics-$(1).csv
bdt_test_metrics_files+=output/bdt-test-metrics-$(1).csv

.PHONY: bdt-$(1)
bdt-$(1): output/bdt-$(1).done

output/bdt-$(1).done: config-bdt/$(1).ini output/pp-$(1).done
	./fit_bdt.py $$< && touch $$@

.PHONY: plot-bdt-train-$(1)
plot-bdt-train-$(1): output/bdt-$(1).done
	./plot_prob_pred.py --mean output/bdt-train-pred-mean-$(1).csv \
	--std output/bdt-train-pred-std-$(1).csv \
	--target output/train-target-$(1).csv

.PHONY: plot-bdt-test-$(1)
plot-bdt-test-$(1): output/bdt-$(1).done
	./plot_prob_pred.py --mean output/bdt-test-pred-mean-$(1).csv \
	--std output/bdt-test-pred-std-$(1).csv \
	--target output/test-target-$(1).csv

output/bdt-timing-$(1).csv: output/bdt-$(1).done

output/bdt-train-metrics-$(1).csv: output/bdt-$(1).done
	./calc_prob_metrics.py --out $$@ \
	--mean output/bdt-train-pred-mean-$(1).csv \
	--std output/bdt-train-pred-std-$(1).csv \
	--target output/train-target-$(1).csv

output/bdt-test-metrics-$(1).csv: output/bdt-$(1).done
	./calc_prob_metrics.py --out $$@ \
	--mean output/bdt-test-pred-mean-$(1).csv \
	--std output/bdt-test-pred-std-$(1).csv \
	--target output/test-target-$(1).csv

output/bdt-train-pred-$(1).csv: output/bdt-$(1).done
	./combine_prob_forecast.py --mean output/bdt-train-pred-mean-$(1).csv \
	--std output/bdt-train-pred-std-$(1).csv \
	--out $$@
	
output/bdt-test-pred-$(1).csv: output/bdt-$(1).done
	./combine_prob_forecast.py --mean output/bdt-test-pred-mean-$(1).csv \
	--std output/bdt-test-pred-std-$(1).csv \
	--out $$@
endef
$(foreach t,$(bdt),$(eval $(call gen_bdt,$(t))))

.PHONY: all-bdt
all-bdt: $(patsubst %,bdt-%,$(bdt)) \
all-bdt-timing all-bdt-train-metrics all-bdt-test-metrics \
$(patsubst %,output/bdt-train-pred-%.csv,$(bdt)) \
$(patsubst %,output/bdt-test-pred-%.csv,$(bdt))

.PHONY: all-bdt-timing
all-bdt-timing: output/bdt-timing.csv

.PHONY: all-bdt-train-metrics
all-bdt-train-metrics: output/bdt-train-metrics.csv

.PHONY: all-bdt-test-metrics
all-bdt-test-metrics: output/bdt-test-metrics.csv

output/bdt-timing.csv: $(bdt_timing_files)
	./gen_report.py --out $@ $^

output/bdt-train-metrics.csv: $(bdt_train_metrics_files)
	./gen_report.py --out $@ $^

output/bdt-test-metrics.csv: $(bdt_test_metrics_files)
	./gen_report.py --out $@ $^


#-------------------------------------------------------------------------------
# OFFICIAL BOOST DECISION TREE
#-------------------------------------------------------------------------------
obdt:=$(patsubst config-obdt/%.ini,%,$(wildcard config-obdt/*.ini))

define gen_obdt
obdt_timing_files+=output/obdt-timing-$(1).csv
obdt_train_metrics_files+=output/obdt-train-metrics-$(1).csv
obdt_test_metrics_files+=output/obdt-test-metrics-$(1).csv

.PHONY: obdt-$(1)
obdt-$(1): output/obdt-$(1).done

output/obdt-$(1).done: config-obdt/$(1).ini output/pp-$(1).done
	./fit_obdt.py $$< && touch $$@

.PHONY: plot-obdt-train-$(1)
plot-obdt-train-$(1): output/obdt-$(1).done
	./plot_prob_pred.py --mean output/obdt-train-pred-mean-$(1).csv \
	--std output/obdt-train-pred-std-$(1).csv \
	--target output/train-target-$(1).csv

.PHONY: plot-obdt-test-$(1)
plot-obdt-test-$(1): output/obdt-$(1).done
	./plot_prob_pred.py --mean output/obdt-test-pred-mean-$(1).csv \
	--std output/obdt-test-pred-std-$(1).csv \
	--target output/test-target-$(1).csv

output/obdt-timing-$(1).csv: output/obdt-$(1).done

output/obdt-train-metrics-$(1).csv: output/obdt-$(1).done
	./calc_prob_metrics.py --out $$@ \
	--mean output/obdt-train-pred-mean-$(1).csv \
	--std output/obdt-train-pred-std-$(1).csv \
	--target output/train-target-$(1).csv

output/obdt-test-metrics-$(1).csv: output/obdt-$(1).done
	./calc_prob_metrics.py --out $$@ \
	--mean output/obdt-test-pred-mean-$(1).csv \
	--std output/obdt-test-pred-std-$(1).csv \
	--target output/test-target-$(1).csv

output/obdt-train-pred-$(1).csv: output/obdt-$(1).done
	./combine_prob_forecast.py --mean output/obdt-train-pred-mean-$(1).csv \
	--std output/obdt-train-pred-std-$(1).csv \
	--out $$@
	
output/obdt-test-pred-$(1).csv: output/obdt-$(1).done
	./combine_prob_forecast.py --mean output/obdt-test-pred-mean-$(1).csv \
	--std output/obdt-test-pred-std-$(1).csv \
	--out $$@
endef
$(foreach t,$(obdt),$(eval $(call gen_obdt,$(t))))

.PHONY: all-obdt
all-obdt: $(patsubst %,obdt-%,$(obdt)) \
all-oobdt-timing all-oobdt-train-metrics all-oobdt-test-metrics \
$(patsubst %,output/obdt-train-pred-%.csv,$(obdt)) \
$(patsubst %,output/obdt-test-pred-%.csv,$(obdt))

.PHONY: all-obdt-timing
all-obdt-timing: output/obdt-timing.csv

.PHONY: all-obdt-train-metrics
all-obdt-train-metrics: output/obdt-train-metrics.csv

.PHONY: all-obdt-test-metrics
all-obdt-test-metrics: output/obdt-test-metrics.csv

output/obdt-timing.csv: $(obdt_timing_files)
	./gen_report.py --out $@ $^

output/obdt-train-metrics.csv: $(obdt_train_metrics_files)
	./gen_report.py --out $@ $^

output/obdt-test-metrics.csv: $(obdt_test_metrics_files)
	./gen_report.py --out $@ $^

#-------------------------------------------------------------------------------
# BOOST FUZZY DECISION TREE
#-------------------------------------------------------------------------------
bfdt:=$(patsubst config-bfdt/%.ini,%,$(wildcard config-bfdt/*.ini))

define gen_bfdt
bfdt_timing_files+=output/bfdt-timing-$(1).csv
bfdt_train_metrics_files+=output/bfdt-train-metrics-$(1).csv
bfdt_test_metrics_files+=output/bfdt-test-metrics-$(1).csv

.PHONY: bfdt-$(1)
bfdt-$(1): output/bfdt-$(1).done

output/bfdt-$(1).done: config-bfdt/$(1).ini output/pp-$(1).done
	./fit_bfdt.py $$< && touch $$@

.PHONY: plot-bfdt-train-$(1)
plot-bfdt-train-$(1): output/bfdt-$(1).done
	./plot_prob_pred.py --mean output/bfdt-train-pred-mean-$(1).csv \
	--std output/bfdt-train-pred-std-$(1).csv \
	--target output/train-target-$(1).csv

.PHONY: plot-bfdt-test-$(1)
plot-bfdt-test-$(1): output/bfdt-$(1).done
	./plot_prob_pred.py --mean output/bfdt-test-pred-mean-$(1).csv \
	--std output/bfdt-test-pred-std-$(1).csv \
	--target output/test-target-$(1).csv

output/bfdt-timing-$(1).csv: output/bfdt-$(1).done

output/bfdt-train-metrics-$(1).csv: output/bfdt-$(1).done
	./calc_prob_metrics.py --out $$@ \
	--mean output/bfdt-train-pred-mean-$(1).csv \
	--std output/bfdt-train-pred-std-$(1).csv \
	--target output/train-target-$(1).csv

output/bfdt-test-metrics-$(1).csv: output/bfdt-$(1).done
	./calc_prob_metrics.py --out $$@ \
	--mean output/bfdt-test-pred-mean-$(1).csv \
	--std output/bfdt-test-pred-std-$(1).csv \
	--target output/test-target-$(1).csv

output/bfdt-train-pred-$(1).csv: output/bfdt-$(1).done
	./combine_prob_forecast.py --mean output/bfdt-train-pred-mean-$(1).csv \
	--std output/bfdt-train-pred-std-$(1).csv \
	--out $$@
	
output/bfdt-test-pred-$(1).csv: output/bfdt-$(1).done
	./combine_prob_forecast.py --mean output/bfdt-test-pred-mean-$(1).csv \
	--std output/bfdt-test-pred-std-$(1).csv \
	--out $$@
endef
$(foreach t,$(bfdt),$(eval $(call gen_bfdt,$(t))))

.PHONY: all-bfdt
all-bfdt: $(patsubst %,bfdt-%,$(bfdt)) \
all-bfdt-timing all-bfdt-train-metrics all-bfdt-test-metrics \
$(patsubst %,output/bfdt-train-pred-%.csv,$(bfdt)) \
$(patsubst %,output/bfdt-test-pred-%.csv,$(bfdt))

.PHONY: all-bfdt-timing
all-bfdt-timing: output/bfdt-timing.csv

.PHONY: all-bfdt-train-metrics
all-bfdt-train-metrics: output/bfdt-train-metrics.csv

.PHONY: all-bfdt-test-metrics
all-bfdt-test-metrics: output/bfdt-test-metrics.csv

output/bfdt-timing.csv: $(bfdt_timing_files)
	./gen_report.py --out $@ $^

output/bfdt-train-metrics.csv: $(bfdt_train_metrics_files)
	./gen_report.py --out $@ $^

output/bfdt-test-metrics.csv: $(bfdt_test_metrics_files)
	./gen_report.py --out $@ $^

