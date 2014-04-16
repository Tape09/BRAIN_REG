#include "itkImageRegistrationMethod.h"
#include "itkAffineTransform.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkGradientDescentOptimizer.h"
#include "itkNormalizeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkEllipseSpatialObject.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

using namespace itk;
using namespace std;

const unsigned int Dimension = 3;
typedef short PixelType;

typedef Image< PixelType, Dimension >  ImageType;
typedef ImageFileReader<ImageType> ReaderType;
typedef ImageFileWriter< ImageType  > WriterType;



int main(int argc, char *argv[]) {
	ReaderType::Pointer fixed_reader = ReaderType::New();
	ReaderType::Pointer moving_reader = ReaderType::New();

	ImageType::Pointer fixed_image = ImageType::New();
	ImageType::Pointer moving_image = ImageType::New();

	fixed_reader->SetFileName("brain0.nii.gz");
	fixed_reader->Update();
	fixed_image = fixed_reader->GetOutput();

	moving_reader->SetFileName("brain1.nii.gz");
	moving_reader->Update();
	moving_image = moving_reader->GetOutput();

	//cout << "Moving:" << endl;
	//cout << moving_image->GetOrigin() << endl;
	//cout << moving_image->GetSpacing() << endl;
	//cout << moving_image->GetDirection() << endl << endl;

	//cout << "fixed:" << endl;
	//cout << fixed_image->GetOrigin() << endl;
	//cout << fixed_image->GetSpacing() << endl;
	//cout << fixed_image->GetDirection() << endl << endl;

	typedef itk::Image< double, Dimension> InternalImageType;
	typedef itk::NormalizeImageFilter<ImageType, InternalImageType> NormalizeFilterType;

	NormalizeFilterType::Pointer fixedNormalizer = NormalizeFilterType::New();
	NormalizeFilterType::Pointer movingNormalizer = NormalizeFilterType::New();

	fixedNormalizer->SetInput(fixed_image);
	movingNormalizer->SetInput(moving_image);

	typedef itk::DiscreteGaussianImageFilter<InternalImageType, InternalImageType> GaussianFilterType;

	GaussianFilterType::Pointer fixedSmoother = GaussianFilterType::New();
	GaussianFilterType::Pointer movingSmoother = GaussianFilterType::New();

	fixedSmoother->SetVariance(2.0);
	movingSmoother->SetVariance(2.0);

	fixedSmoother->SetInput(fixedNormalizer->GetOutput());
	movingSmoother->SetInput(movingNormalizer->GetOutput());

	typedef itk::AffineTransform< double, Dimension > TransformType;
	typedef itk::GradientDescentOptimizer OptimizerType;
	typedef itk::LinearInterpolateImageFunction<InternalImageType, double> InterpolatorType;
	typedef itk::ImageRegistrationMethod<InternalImageType,	InternalImageType > RegistrationType;
	typedef itk::MutualInformationImageToImageMetric<InternalImageType,	InternalImageType > MetricType;

	TransformType::Pointer transform = TransformType::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	RegistrationType::Pointer registration = RegistrationType::New();
	MetricType::Pointer metric = MetricType::New();

	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);	
	registration->SetMetric(metric);

	metric->SetFixedImageStandardDeviation(0.4);
	metric->SetMovingImageStandardDeviation(0.4);

	fixedSmoother->GetOutput()->SetOrigin(fixed_image->GetOrigin());
	fixedSmoother->GetOutput()->SetSpacing(fixed_image->GetSpacing());
	fixedSmoother->GetOutput()->SetDirection(fixed_image->GetDirection());
	movingSmoother->GetOutput()->SetOrigin(moving_image->GetOrigin());
	movingSmoother->GetOutput()->SetSpacing(moving_image->GetSpacing());
	movingSmoother->GetOutput()->SetDirection(moving_image->GetDirection());

	registration->SetFixedImage(fixedSmoother->GetOutput());
	registration->SetMovingImage(movingSmoother->GetOutput());

	fixedNormalizer->Update();

	ImageType::RegionType fixedImageRegion = fixedNormalizer->GetOutput()->GetBufferedRegion();
	registration->SetFixedImageRegion(fixedImageRegion);

	typedef RegistrationType::ParametersType ParametersType;
	ParametersType initialParameters(transform->GetNumberOfParameters());

	cout << "NParam= " << transform->GetNumberOfParameters() << endl; //12

	// rotation matrix (identity)
	initialParameters[0] = 1.0; 
	initialParameters[1] = 0.0; 
	initialParameters[2] = 0.0; 
	initialParameters[3] = 0.0;
	initialParameters[4] = 1.0;
	initialParameters[5] = 0.0;
	initialParameters[6] = 0.0;
	initialParameters[7] = 0.0;
	initialParameters[8] = 1.0;

	// translation vector
	initialParameters[9] = 0.0;
	initialParameters[10] = 0.0;
	initialParameters[11] = 0.0;

	registration->SetInitialTransformParameters(initialParameters);

	const unsigned int numberOfPixels = fixedImageRegion.GetNumberOfPixels();
	const unsigned int numberOfSamples = static_cast<unsigned int>(numberOfPixels * 0.01);

	metric->SetNumberOfSpatialSamples(numberOfSamples);

	optimizer->SetLearningRate(1.0);
	optimizer->SetNumberOfIterations(1);
	optimizer->MaximizeOn(); 	

	//cout << "Moving:" << endl;
	//cout << registration->GetMovingImage()->GetOrigin() << endl;
	//cout << registration->GetMovingImage()->GetDirection() << endl;
	//cout << registration->GetMovingImage()->GetSpacing() << endl << endl;

	//cout << "fixed:" << endl;
	//cout << registration->GetFixedImage()->GetOrigin() << endl;
	//cout << registration->GetFixedImage()->GetDirection() << endl;
	//cout << registration->GetFixedImage()->GetSpacing() << endl << endl;

	try {
		registration->Update();
		std::cout << "Optimizer stop condition: "
			<< registration->GetOptimizer()->GetStopConditionDescription()
			<< std::endl;
	} catch (itk::ExceptionObject & err) {
		std::cout << "ExceptionObject caught !" << std::endl;
		std::cout << err << std::endl;
		return EXIT_FAILURE;
	}

	ParametersType finalParameters = registration->GetLastTransformParameters();

	std::cout << "Final Parameters: " << finalParameters << std::endl;

	unsigned int numberOfIterations = optimizer->GetCurrentIteration();

	double bestValue = optimizer->GetValue();

	// Print out results
	std::cout << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " Iterations    = " << numberOfIterations << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;
	std::cout << " Numb. Samples = " << numberOfSamples << std::endl;

	typedef itk::ResampleImageFilter<ImageType, ImageType > ResampleFilterType;

	TransformType::Pointer finalTransform = TransformType::New();

	finalTransform->SetParameters(finalParameters);
	finalTransform->SetFixedParameters(transform->GetFixedParameters());

	ResampleFilterType::Pointer resample = ResampleFilterType::New();

	resample->SetTransform(finalTransform);
	resample->SetInput(moving_image);

	resample->SetSize(fixed_image->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(fixed_image->GetOrigin());
	resample->SetOutputSpacing(fixed_image->GetSpacing());
	resample->SetOutputDirection(fixed_image->GetDirection());
	//resample->SetDefaultPixelValue(100);

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName("output.nii.gz");
	writer->SetInput(resample->GetOutput());
	writer->Update();

	return EXIT_SUCCESS;
}