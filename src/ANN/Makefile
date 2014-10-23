#
# Linux makefile for ANN library
#



#
# List of files
#

CCSRCS = ANN.cpp brute.cpp kd_tree.cpp kd_util.cpp kd_split.cpp \
	 kd_dump.cpp kd_search.cpp kd_pr_search.cpp kd_fix_rad_search.cpp \
	 bd_tree.cpp bd_search.cpp bd_pr_search.cpp bd_fix_rad_search.cpp \
	 perf.cpp
INCS =   kd_tree.h kd_split.h kd_util.h kd_search.h \
	 kd_pr_search.h kd_fix_rad_search.h perf.h pr_queue.h pr_queue_k.h
OBJS = $(CCSRCS:.cpp=.o)



#
# Options
#

CC=g++
CPPFLAGS=-Wall -g -I.
LDFLAGS=-g



#
# Output library name
#

LIB=./libANN.a



#
# Make targets
#

$(LIB): $(OBJS)
	    - rm -f $(LIB)
	    ar ur $(LIB) $(OBJS)

test: test.o $(LIB) 
	    $(CC) -o test $(CPPFLAGS) $(LDFLAGS) test.o $(LIB)

clean:
	    -  rm -f *~ *.o *.obj $(LIB) test test.exe
