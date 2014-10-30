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

// Program arguments

static char *A_filename = NULL;
static char *Ap_filename = NULL;
static char *B_filename = NULL;
static char *Bp_filename = NULL;
static char *Bs_filename = NULL;
static R2Pixel mask_color = R2blue_pixel;
static int neighborhood_size = 5;

////////////////////////////////////////////////////////////////////////////////

struct Location {
  int i; int j;
};

bool OutOfBounds(int i, int j, const R2Image *image)
{
  return i < 0 || j < 0 || i >= image->Width() || j >= image->Height();
}

bool OutOfBounds(int i, int j, int width, int height, int boundary = 0)
{
  return i < boundary || j < boundary
      || i >= width - boundary
      || j >= height - boundary;
}

static Location
GetLocationFromIndex(int index, int width, int height, int boundary = 0) {
  Location l;
  if (index == -1) {
    l.i = -1;
    l.j = -1;
    return l;
  }

  int subHeight = height - boundary * 2;
  l.i = (int) (index / subHeight) + boundary;
  l.j = index % subHeight + boundary;
  return l;
}

static int
GetIndexFromLocation(int i, int j, int width, int height, int boundary = 0) {
  if (OutOfBounds(i, j, width, height, boundary)) {
    return -1;
  }

  int subI = i - boundary;
  int subJ = j - boundary;
  int subHeight = height - boundary * 2;

  return subI * subHeight + subJ;
}

double GetFeatureDist(ANNpoint v1, ANNpoint v2, int dim)
{
  double dist = 0;
  for (int i = 0; i < dim; i ++) {
    dist += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }
  return dist;
}

double GetLuminosity(int i, int j, const R2Image *image)
{
  return image->Pixel(i, j).Y();
}

double Gauss(int dI, int dJ, int regionSize)
{
  double distSq = dI * dI + dJ * dJ;
  double sigmaSq = regionSize * 0.4;
  double gauss = exp(- distSq / sigmaSq); // not scaled correctly
  return gauss;
}

static double
CalculateMeanLuminance(const R2Image *A)
{
  double lumSum = 0;
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      lumSum += GetLuminosity(aI, aJ, A);
    }
  }

  return lumSum / (A->Width() * A->Height());
}

static double
CalculateStdDev(const R2Image *A, double mean)
{
  double variance = 0;
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      double diff = GetLuminosity(aI, aJ, A) - mean;
      variance += diff * diff;
    }
  }

  return sqrt(variance / (A->Width() * A->Height()));
}

static void
RemapLuminance(R2Image *A, double meanA, double stdDevA, double meanB, double stdDevB)
{
  printf("%f, %f, %f, %f\n", meanA, stdDevA, meanB, stdDevB);
  for (int aI = 0; aI < A->Width(); aI++) {
    for (int aJ = 0; aJ < A->Height(); aJ++) {
      R2Pixel o = A->Pixel(aI, aJ);
      double newY = (stdDevB / stdDevA) * (o.Y() - meanA) + meanB;

      R2Pixel p;
      p.SetYIQ(newY, o.I(), o.Q());
      A->SetPixel(aI, aJ, p);
    }
  }
}

static void
RemapLuminance(R2Image *A, R2Image *Ap, const R2Image *B)
{
  double meanA = CalculateMeanLuminance(A);
  double meanB = CalculateMeanLuminance(B);

  double stdDevA = CalculateStdDev(A, meanA);
  double stdDevB = CalculateStdDev(B, meanB);

  RemapLuminance(A,  meanA, stdDevA, meanB, stdDevB);
  RemapLuminance(Ap, meanA, stdDevA, meanB, stdDevB);
}

static void
FillFeature(double *output, int aI, int aJ, const R2Image *A, int regionSize, int dimension)
{
  for (int regionI = 0; regionI < regionSize; regionI ++) {
    for (int regionJ = 0; regionJ < regionSize; regionJ ++) {
      int regionIndex = regionI * regionSize + regionJ;
      int dI = regionI - regionSize / 2;
      int dJ = regionJ - regionSize / 2;

      // Skip if the relevant pixel doesn't exist yet.
      if (regionIndex >= dimension) {
        continue;
      }

      // Skip if this is out of bounds
      if (OutOfBounds(aI + dI, aJ + dJ, A)) {
        output[regionIndex] = 0;
        continue;
      }

      output[regionIndex] = GetLuminosity(aI + dI, aJ + dJ, A) * Gauss(dI, dJ, regionSize);
    }
  }
}

static void
FillFeature(double *output, int aI, int aJ, const R2Image *A, const R2Image *Ap,
           int region, int dim1, int dim2)
{
  FillFeature(&output[0], aI, aJ, A,  region, dim1);
  FillFeature(&output[dim1], aI, aJ, Ap, region, dim2);
}

////////////////////////////////////////////////////////////////////////////////
// Brute force methods

static int
FindBestMatchBrute(ANNpointArray featuresA, int numA,
                   ANNpoint featureB, int dim) {
  double minDist = DBL_MAX;
  double minIndex;

  for (int i = 0; i < numA; i ++) {
    double dist = GetFeatureDist(featuresA[i], featureB, dim);
    if (dist < minDist) {
      minDist = dist;
      minIndex = i;
    }
  }
  return minIndex;
}

static void
RunAnalogyBrute(const R2Image *A, const R2Image *Ap,
                const R2Image *B, R2Image *Bp, Location **sources)
{
  int regionSize = 5;

  int width  = A->Width();
  int height = A->Height();

  // Dimensions of feature vectors & their constituents
  int dimA  = regionSize * regionSize;
  int dimAp = floor(dimA * 0.5);
  int dim   = dimA + dimAp;

  // Only look at a subsection of A
  int boundaryA = (int) (regionSize / 2);
  int numA = (A->Width()  - 2 * boundaryA)
           * (A->Height() - 2 * boundaryA);

  auto FillFeatureA = [&](double *output, int aI, int aJ) {
    return FillFeature(output, aI, aJ, A, Ap, regionSize, dimA, dimAp);
  };

  auto FillFeatureB = [&](double *output, int bI, int bJ) {
    return FillFeature(output, bI, bJ, B, Bp, regionSize, dimA, dimAp);
  };

  ANNpointArray featuresA = annAllocPts(numA, dim);
  for (int i = 0; i < numA; i ++) {
    Location a = GetLocationFromIndex(i, width, height, boundaryA);
    FillFeatureA(featuresA[i], a.i, a.j);
  }

  ANNpoint featureB = annAllocPt(dim);

  for (int bI = 0; bI < B->Width(); bI++) {
    printf("%d out of %d\n", bI, B->Width());
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      FillFeatureB(featureB, bI, bJ);

      int indexA = FindBestMatchBrute(featuresA, numA, featureB, dim);
      Location a = GetLocationFromIndex(indexA, width, height, boundaryA);
      int aI = a.i;
      int aJ = a.j;

      R2Pixel pixelB = B->Pixel(bI, bJ);
      R2Pixel pixelAp = Ap->Pixel(aI, aJ);

      R2Pixel p;
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);

      sources[bI][bJ] = a;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// ANN methods

static int
ANNMatch(ANNpoint queryPoint, ANNkd_tree *kdTree)
{
  int numNN = 1;
  ANNidx nnIdx;
  ANNdist nnDist;
  int errorBound = 0;

  kdTree->annkSearch(queryPoint, numNN, &nnIdx, &nnDist, errorBound);

  return nnIdx;
}

static Location *
CoherencePossibleSources(int bI, int bJ, int regionSize,
                         int width, int height, int boundaryA, int boundaryB,
                         Location **sources, bool skipUnrendered = true)
{
  int numPossibleSources = regionSize * regionSize;
  Location *possibleSources = new Location[numPossibleSources];
  for (int i = 0; i < numPossibleSources; i ++) {
    possibleSources[i].i = -1;
    possibleSources[i].j = -1;
  }

  for (int regionI = 0; regionI < regionSize; regionI ++) {
    for (int regionJ = 0; regionJ < regionSize; regionJ ++) {
      int regionIndex = regionI * regionSize + regionJ;
      int dI = regionI - regionSize / 2;
      int dJ = regionJ - regionSize / 2;

      if (skipUnrendered) {
        if (dJ > 0) break;
        if (dJ == 0 && dI >= 0) break;
      }

      if (OutOfBounds(bI + dI, bJ + dJ, width, height, boundaryB)) {
        continue;
      }

      Location a = sources[bI + dI][bJ + dJ];
      a.i -= dI;
      a.j -= dJ;

      if (OutOfBounds(a.i, a.j, width, height, boundaryA)) {
        continue;
      }

      possibleSources[regionIndex] = a;
    }
  }

  return possibleSources;
}

template<typename Func>
static int
CoherenceMatch(int bI, int bJ, ANNpointArray featuresA,
               Func FillFeatureB,
               int width, int height, int boundaryA, int boundaryB,
               int regionSize, int dim, Location **sources)
{
  double minDist = DBL_MAX;
  int minIndexA = -1;

  // Compare coherence sources to this feature in B
  ANNpoint featureB = annAllocPt(dim);
  FillFeatureB(featureB, bI, bJ);

  // Get the possible coherence sources
  int numLocations = regionSize * regionSize;
  Location *locationsA = CoherencePossibleSources(
    bI, bJ, regionSize, width, height, boundaryA, boundaryB, sources);

  for (int i = 0; i < numLocations; i ++) {
    Location a = locationsA[i];
    int indexA = GetIndexFromLocation(a.i, a.j, width, height, boundaryA);
    if (indexA == -1) continue;

    ANNpoint featureA = featuresA[indexA];

    double dist = GetFeatureDist(featureA, featureB, dim);
    if (dist < minDist) {
      minDist = dist;
      minIndexA = indexA;
    }
  }

  return minIndexA;
}

static int
ChooseBestMatch(int annIndex, int cohIndex, double coherence,
                ANNpointArray featuresA, ANNpoint featureB, int dim)
{
  if (cohIndex == -1) {
    return annIndex;
  }

  ANNpoint annFeature = featuresA[annIndex];
  ANNpoint cohFeature = featuresA[cohIndex];

  double annDist = GetFeatureDist(annFeature, featureB, dim);
  double cohDist = GetFeatureDist(cohFeature, featureB, dim);

  if (cohDist <= annDist * (1 + pow(2, 0) * coherence)) {
    return cohIndex;
  } else {
    return annIndex;
  }
}

static void
RunAnalogyANN(const R2Image *A, const R2Image *Ap,
              const R2Image *B, R2Image *Bp,
              double coherence, Location **sources)
{
  int regionSize = 5;

  // Dimensions of feature vectors & their constituents
  int dimA  = regionSize * regionSize;
  int dimAp = floor(dimA * 0.5);
  int dim   = dimA + dimAp;

  int width = A->Width();
  int height = A->Height();

  // Only look at a subsection of A
  int boundaryA = (int) (regionSize / 2);
  int numA = (A->Width()  - 2 * boundaryA)
           * (A->Height() - 2 * boundaryA);

  auto FillFeatureA = [&](double *output, int aI, int aJ) {
    return FillFeature(output, aI, aJ, A, Ap, regionSize, dimA, dimAp);
  };

  auto FillFeatureB = [&](double *output, int bI, int bJ) {
    return FillFeature(output, bI, bJ, B, Bp, regionSize, dimA, dimAp);
  };

  ANNpointArray featuresA = annAllocPts(numA, dim);
  for (int i = 0; i < numA; i ++) {
    Location a = GetLocationFromIndex(i, width, height, boundaryA);
    FillFeatureA(featuresA[i], a.i, a.j);
  }

  ANNkd_tree *treeA = new ANNkd_tree(featuresA, numA, dim);

  printf("Generating Bp\n");
  ANNpoint featureB = annAllocPt(dim);
  for (int   bI = 0; bI < width;  bI++) {
    for (int bJ = 0; bJ < height; bJ++) {
      FillFeatureB(featureB, bI, bJ);

      int annIndex = ANNMatch(featureB, treeA);
      int cohIndex = CoherenceMatch(bI, bJ, featuresA, FillFeatureB,
                                    width, height, boundaryA, 0,
                                    regionSize, dim, sources);

      int index = ChooseBestMatch(annIndex, cohIndex, coherence,
                                  featuresA, featureB, dim);
      Location a = GetLocationFromIndex(index, width, height, boundaryA);
      sources[bI][bJ] = a;

      R2Pixel p;
      R2Pixel pixelAp = Ap->Pixel(a.i, a.j);
      R2Pixel pixelB = B->Pixel(bI, bJ);
      p.SetYIQ(pixelAp.Y(), pixelB.I(), pixelB.Q());
      Bp->SetPixel(bI, bJ, p);
    }
    printf("%d out of %d\n", bI, width);
  }

  annDeallocPts(featuresA);
  annDeallocPt(featureB);
  delete treeA;
}

////////////////////////////////////////////////////////////////////////////////
// Multi resolution

static R2Image *
CreateLowerResImage(const R2Image *A)
{
  // Create a lower res, blurred image
  R2Image *B = new R2Image(A->Width() / 2, A->Height() / 2);

  for (int bI = 0; bI < B->Width(); bI++) {
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      int aI = (int) ((bI + 0.5) * 2);
      int aJ = (int) ((bJ + 0.5) * 2);

      double lumSum = 0;
      double iSum = 0;
      double qSum = 0;
      double gaussSum = 0;

      int radius = 2;
      for (int dI = -radius; dI <= radius; dI ++) {
        for (int dJ = -radius; dJ <= radius; dJ ++) {
          if (OutOfBounds(aI + dI, aJ + dJ, A)) {
            continue;
          }

          double gauss = Gauss(dI, dJ, 2 * radius);
          R2Pixel a = A->Pixel(aI + dI, aJ + dJ);
          lumSum += a.Y() * gauss;
          iSum   += a.I() * gauss;
          qSum   += a.Q() * gauss;
          gaussSum += gauss;
        }
      }

      R2Pixel b;
      b.SetYIQ(lumSum / gaussSum, iSum / gaussSum, qSum / gaussSum);
      B->SetPixel(bI, bJ, b);
    }
  }

  return B;
}

static R2Image **
CreateMultiResImages(const R2Image *A, int levels)
{
  R2Image **As  = new R2Image *[levels];
  As[levels - 1] = new R2Image(*A);

  for (int i = levels - 2; i >= 0; i --) {
    As[i] = CreateLowerResImage(As[i + 1]);
  }

  return As;
}

static void
RunAnalogyMulti(const R2Image *A, const R2Image *Ap,
                const R2Image *B, const R2Image *Bp,
                double coherence, Location **sources,
                int levels)
{
  R2Image **As  = CreateMultiResImages(A, levels);
  R2Image **Aps = CreateMultiResImages(Ap, levels);
  R2Image **Bs  = CreateMultiResImages(B, levels);
  R2Image **Bps = CreateMultiResImages(Bp, levels);

  // for (int i = 0; i < levels; i ++) {
  //   char *filename = new char[20];
  //   strcpy(filename, Bp_filename);
  //   strstr(filename, ".bmp")[0] = '\0';
  //   strcat(filename, ".x.bmp");
  //   strstr(filename, ".x.bmp")[1] = '0' + i;
  //   printf("%s\n", filename);
  //   Bs[i]->Write(filename);
  // }

  for (int level = 0; level < levels; level ++) {

  }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper code

static R2Image *
CreateAnalogyImage(const R2Image *A, const R2Image *Ap, const R2Image *B)
{
  // Allocate Bp image
  R2Image *Bp = new R2Image(B->Width(), B->Height());
  if (!Bp) {
    fprintf(stderr, "Unable to allocate output_image\n");
    exit(-1);
  }

  R2Image *newA  = new R2Image(*A);
  R2Image *newAp = new R2Image(*Ap);
  RemapLuminance(newA, newAp, B);

  Location **sources = new Location *[Bp->Width()];
  for (int i = 0; i < Bp->Width(); i ++) {
    sources[i] = new Location[Bp->Height()];
  }

  // RunAnalogyBrute(newA, newAp, B, Bp, sources);
  RunAnalogyANN(newA, newAp, B, Bp, 0.2, sources);
  // RunAnalogyMulti(newA, newAp, B, Bp, 0.2, sources, 4);

  R2Image *sourcesImage = new R2Image(*Bp);
  for (int bI = 0; bI < B->Width(); bI++) {
    for (int bJ = 0; bJ < B->Height(); bJ++) {
      Location a = sources[bI][bJ];
      double aI = a.i;
      double aJ = a.j;

      R2Pixel p;
      p.SetRed(aI / B->Width());
      p.SetGreen(aJ / B->Height());
      sourcesImage->SetPixel(bI, bJ, p);
    }
  }

  sourcesImage->Write(Bs_filename);

  delete sourcesImage;
  delete newA;
  delete newAp;
  for (int i = 0; i < Bp->Width(); i ++) delete sources[i];
  delete sources;

  // Return Bp image
  return Bp;
}



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

  Bs_filename = new char[strlen(Bp_filename) + 4];
  strcpy(Bs_filename, Bp_filename);
  strstr(Bs_filename, ".bmp")[0] = '\0';
  strcat(Bs_filename, ".src.bmp");
  printf("%s\n", Bs_filename);

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
