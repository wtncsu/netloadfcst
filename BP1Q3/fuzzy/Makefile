.DEFAULT_GOAL:=all

output_dir:=output
data_files:=$(wildcard data/*/*.csv)
result_files:=$(patsubst data/%.csv,$(output_dir)/%/result,$(data_files))

$(output_dir)/%/result: data/%.csv data/%.ini
	mkdir -p $(output_dir)/$(*)
	./fingers_crossed.py \
		--csv data/$(*).csv \
		--config data/$(*).ini \
		--output $(output_dir)/$(*) 2> $(output_dir)/$(*)/log
	@echo finished $(@)

$(output_dir):
	mkdir -p $(output_dir)


.PHONY: all
all: $(result_files)
	echo $(result_files)

.PHONY: clean
clean:
	rm -rf $(output_dir)

