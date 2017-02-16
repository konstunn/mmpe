
R_OPTS=--vanilla

.PHONY: main.tex

main.pdf: main.tex Makefile
	latexmk -pdf $<

main.tex: main.Rnw
	R ${R_OPTS} \
		-e "library(knitr)" \
		-e "knit('$<')"

view: main.pdf
	xdg-open $< > /dev/null 2>&1 &

clean:
	latexmk -f -C main.pdf
	rm -f main.tex
