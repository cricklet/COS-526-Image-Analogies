This directory contains skeleton code as a starting point for 
assignment1 of COS 526. 


FILE STRUCTURE
==============

There are several files, but you should mainly change analogy.cpp.

  anaology,cpp - Main program, parses the command line arguments, 
    and executes the image analogy algorithm
  R2Image.[cpp/h] - Image class useful for reading and writing images
  R2Pixel.[cpp/h] - Pixel class used by R2Image class
  ANN/ - A library for approximate nearest neighbor search
  jpeg/ - A library for reading/writing JPEG files
  analogy.[vcproj/sln/suo] - Project file for Visual Studio 2005 on Windows
  Makefile - Unix/Mac makefile for building the project with "make". 


COMPILATION
===========

If you are developing on a Windows machine and have Visual Studio
installed, use the provided project solution file (analogy.sln) to
build the program. If you are developing on a Mac or Linux machine,
type "make". In either case, an executable called analogy (or
anaology.exe) will be created in top-level directory.
