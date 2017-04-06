
R_OPTS=--vanilla

.PHONY: main.tex

all: main.pdf

%.pdf: %.tex Makefile
	latexmk -pdf $<

%.tex: %.Rnw Makefile
	R ${R_OPTS} \
		-e "library(knitr)" \
		-e "knit('$<')"

view: main.pdf
	xdg-open $< > /dev/null 2>&1 &

clean:
	latexmk -f -C main.pdf
	rm -f main.tex
