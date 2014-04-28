IMAGES=

default: main.pdf

.PHONY: fast
fast:
	pdflatex main

main.dvi: */*.tex *.tex Makefile img/* *.bib $(IMAGES)
	latex main
	biber main
	latex main
	latex main

main.ps: main.dvi
	dvips main.dvi

quietps: *.tex Makefile img/* $(IMAGES)
	latex -interaction=batchmode main
	latex -interaction=batchmode main
	latex -interaction=batchmode main
	dvips main.dvi

dvi: main.dvi

ps: main.ps

main.pdf: */*.tex *.tex Makefile img/* *.bib $(IMAGES)
	pdflatex main
	biber main
	pdflatex main
	pdflatex main

pdf: main.pdf

html: dvi 
	for i in * ; do if [ ! -d "$i"] ; then cp "$i" html ; fi ; done
	cd html ; latex2html -html_version 4.0 -no_navigation -no_subdir -info 0 main.tex ; cd ..

clean: 
	-rm -f *.{log,aux}

dist-clean:
	-rm -f *.{log,aux,dvi,ps,pdf,toc,bbl,blg,slo,srs,out,bak,lot,lof}
	-(cd img; make dist-clean)

backup: 
	tar --create --force-local -zf zaloha/knizka-`date +%Y-%m-%d-%H\:%M`.tar.gz `ls -p| egrep -v /$ ` images/* code/*

all: pdf


booklet: main.ps
	cat main.ps | psbook | psnup -2 >main-booklet.ps

img/%.tikz.tex: img/%.gnu
	(cd img; gnuplot $*.gnu)

.PHONY: img
img:
	(cd img; make all)

.PHONY: prepare_upload
prepare_upload: dist-clean img main.pdf
	cp main.pdf `date '+krypto_%y-%m-%d.pdf'`

.PHONY: upload
upload: main.pdf
	scp main.pdf ksp.sk:public_html/fmfi/diplomovka/current.pdf
