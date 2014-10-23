// COS 526, Spring 2010, Assignment 1



// Include files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "R2Pixel.h"
#include "R2Image.h"



// Program arguments

static char *A_filename = NULL;
static char *Ap_filename = NULL;
static char *B_filename = NULL;
static char *Bp_filename = NULL;
static R2Pixel mask_color = R2blue_pixel;
static int neighborhood_size = 5;



static R2Image *
ReadImage(const char *filename) 
{
  // Allocate input_image
  R2Image *image = new R2Image();
  if (!image) {
    fprintf(stderr, "Unable to allocate image\n");
    return NULL;
  }

  // Read input image
  if (!image->Read(filename)) {
    fprintf(stderr, "Unable to read image from %s\n", filename);
    return NULL;
  }

  // Return image
  return image;
}



static R2Image *
CreateAnalogyImage(const R2Image *A, const R2Image *Ap, const R2Image *B)
{
  // Allocate Bp image
  R2Image *Bp = new R2Image(B->Width(), B->Height());
  if (!Bp) {
    fprintf(stderr, "Unable to allocate output_image\n");
    exit(-1);
  }

  // REPLACE CODE STARTING HERE

  // For now, just copy pixels from B
  for (int i = 0; i < B->Width(); i++) {
    for (int j = 0; j < B->Height(); j++) {
      R2Pixel pixel = B->Pixel(i, j);
      Bp->SetPixel(i, j, pixel);
    }
  }

  // REPLACE CODE ENDING HERE

  // Return Bp image
  return Bp;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Parse program arguments 
  argv++, argc--; 
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-neighborhood_size")) { argv++; argc--; neighborhood_size = atoi(*argv); }
      else if (!strcmp(*argv, "-mask_color")) { mask_color.Reset(atof(argv[1]),atof(argv[2]),atof(argv[3]),1); argv+=3; argc-=3; }
      else { fprintf(stderr, "Invalid option: %s\n", *argv); return 0; }
    }
    else {
      if (!A_filename) A_filename = *argv;
      else if (!Ap_filename) Ap_filename = *argv;
      else if (!B_filename) B_filename = *argv;
      else if (!Bp_filename) Bp_filename = *argv;
      else { fprintf(stderr, "Invalid option: %s\n", *argv); return 0; }
    }
    argv++, argc--; 
  }

  // Check program arguments
  if (!A_filename || !Ap_filename || !B_filename || !Bp_filename) {
    fprintf(stderr, "Usage: analogy A A\' B B\'[-mask_color r g b] [-neighborhood_size npixels]\n");
    return 0;
  }

  // Return success
  return 1;
}



int 
main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read input images
  R2Image *A = ReadImage(A_filename);
  if (!A) exit(-1);
  R2Image *Ap = ReadImage(Ap_filename);
  if (!Ap) exit(-1);
  R2Image *B = ReadImage(B_filename);
  if (!B) exit(-1);

  // Create output image
  R2Image *Bp = CreateAnalogyImage(A, Ap, B);
  if (!Bp) exit(-1);

  // Write output image
  if (!Bp->Write(Bp_filename)) exit(-1);

  // Return success
  return 0;
}



