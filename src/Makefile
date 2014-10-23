#
# Unix/Linux makefile for assignment #1
#



# 
# List of source files
#

ANALOGY_SRCS=analogy.cpp R2Image.cpp R2Pixel.cpp
ANALOGY_OBJS=$(ANALOGY_SRCS:.cpp=.o)



#
# Compile and link options
#

CC=g++
CPPFLAGS=-Wall -I. -g
LDFLAGS=-g



#
# Libraries
#

LIBS=jpeg/libjpeg.a ANN/libANN.a



#
# Compile command
#

%.o: %.cpp R2Image.h R2Pixel.h
	    $(CC) $(CPPFLAGS) -c $< -o $@



#
# GNU Make: targets that don't build files
#

.PHONY: all clean distclean



#
# Make targets
#

all: $(LIBS) analogy

analogy: $(LIBS) $(ANALOGY_OBJS) 
	    $(CC) -o analogy $(CPPFLAGS) $(LDFLAGS) $(ANALOGY_OBJS) $(LIBS) -lm

jpeg/libjpeg.a: 
	    cd jpeg; make

ANN/libANN.a: 
	    cd ANN; make

clean:
	    ${RM} -f */*.a */*/*.a *.o */*.o */*/*.o analogy analogy.exe $(LIBS)

distclean:  clean
	    ${RM} -f *~ 






