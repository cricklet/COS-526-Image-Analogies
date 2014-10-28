// COS 526, Spring 2010, Assignment 1



// Include files

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <climits>
#include <ANN/ANN.h>
#include "R2Pixel.h"
#include "R2Image.h"

struct Location {
  int i; int j;
};

bool OutOfBounds(int i, int j, const R2Image *image)
{
  return i >= 0 && j >= 0 && i < image->Width() && j < image->Height();
}

double GetLuminosityFromRegion(
  int dI, int dJ,
  int centerI, int centerJ,
  const R2Image *image)
{
  int imageI = centerI + dI;
  int imageJ = centerJ + dJ;

  return image->Pixel(imageI, imageJ).Y();
}

double CalculateDistance(int aI, int aJ, int bI, int bJ,
  const R2Image *A, const R2Image *Ap, const R2Image *B, R2Image *Bp, int regionSize)
{
  double distance = 0;

  for (int regionI = 0; regionI < regionSize; regionI ++) {
    for (int regionJ = 0; regionJ < regionSize; regionJ ++) {
      int dI = regionI - regionSize / 2;
      int dJ = regionJ - regionSize / 2;

      // Skip if this is out of bounds
      if (OutOfBounds(aI + dI, aJ + dJ, A) ||
          OutOfBounds(bI + dI, bJ + dJ, B)) {
         continue;
      }

      // Get all relevant values
      double valA  = GetLuminosityFromRegion(dI, dJ, aI, aJ, A);
      double valAp = GetLuminosityFromRegion(dI, dJ, aI, aJ, Ap);
      double valB  = GetLuminosityFromRegion(dI, dJ, bI, bJ, B);
      double valBp = GetLuminosityFromRegion(dI, dJ, bI, bJ, Bp);

      // Take a squared difference of them
      distance += (valB - valA) * (valB - valA);

      // Skip Bp if the relevant pixel doesn't exist yet.
      if (dJ > 0) continue;
      if (dJ == 0 && dI >= 0) continue;

      distance += (valBp - valAp) * (valBp - valAp);
    }
  }

  return distance;
}

Location FindBestMatch(int bI, int bJ, const R2Image *A, const R2Image *Ap, const R2Image *B, R2Image *Bp) {
  int region = 3;
  Location p;

  double minDist = INT_MAX;
  int minAI = -1;
  int minAJ = -1;

  // Loop through A, storing the best match aI, aJ for region bI, bJ
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      double dist = CalculateDistance(aI, aJ, bI, bJ, A, Ap, B, Bp, region);
      if (dist < minDist) {
        minAI = aI;
        minAJ = aJ;
      }
    }
  }

  p.i = minAI;
  p.j = minAJ;

  return p;
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

  for (int bI = 0; bI < B->Width(); bI++) {
    printf("%d out of %d\n", bI, B->Width());
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      Location a = FindBestMatch(bI, bJ, A, Ap, B, Bp);
      int aI = a.i;
      int aJ = a.j;

      R2Pixel pixelB = B->Pixel(bI, bJ);
      R2Pixel pixelAp = Ap->Pixel(aI, aJ);

      R2Pixel p;
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);
    }
  }

  // REPLACE CODE ENDING HERE

  // Return Bp image
  return Bp;
}



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
