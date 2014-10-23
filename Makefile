########################################################################
# This Makefile can be used to execute your program once it is compiled.
# It should contain the commands that generate all of the images
# in your submitted output directory from the ones in your input directory.
########################################################################



########################################################################
# Executable name
########################################################################

EXE=src/analogy



########################################################################
# This command tells the make file how to execute src/analogy
# to produce output/foo.jpg from input/foo.A.bmp, input/foo.Ap.bmp,
# and input/foo.B.bmp 
########################################################################

output/%.bmp: input/%.A.bmp input/%.Ap.bmp input/%.B.bmp
	$(EXE) $+ $@

output/%.jpg: output/%.bmp 
	$(EXE) $+ $@



########################################################################
# "make all" or simply "make" should make all output images
########################################################################

all: \
	$(EXE) \
        output/blur.bmp  \
	output/emboss.bmp \
	output/synth1.bmp
	
clean:
	cd src; make clean
	rm -f output/*

$(EXE): src
	cd src; make



