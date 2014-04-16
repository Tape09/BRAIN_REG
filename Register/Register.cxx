#include "itkResampleImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"
#include "itkDirectory.h"
#include "itkFileTools.h"
#include "itkExceptionObject.h"
#include "itkSimilarity3DTransform.h"
#include "itkImageRegistrationMethod.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNormalizeImageFilter.h"
#include "itkAffineTransform.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkCenteredTransformInitializer.h"
#include "itkCommand.h"
#include <iostream>
#include <string>


typedef itk::Image< double, 3 > ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;
typedef itk::AffineTransform<double, 3> TransformType;
//typedef itk::Similarity3DTransform<double> TransformType;
typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
typedef itk::MutualInformationImageToImageMetric<ImageType, ImageType> MetricType;
typedef itk::LinearInterpolateImageFunction<ImageType, double> InterpolatorType;
typedef itk::ImageRegistrationMethod<ImageType, ImageType> RegistrationType;
//typedef itk::DiscreteGaussianImageFilter<double, double> GaussianFilterType;
//typedef itk::MeanSquaresImageToImageMetric<ImageType, ImageType> MetricType;

typedef RegistrationType::ParametersType ParametersType;


using namespace std;


class CommandIterationUpdate : public itk::Command {
public:
	typedef  CommandIterationUpdate   Self;
	typedef  itk::Command             Superclass;
	typedef itk::SmartPointer<Self>   Pointer;
	itkNewMacro(Self);

protected:
	CommandIterationUpdate() {};

public:
	typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
	typedef   const OptimizerType *                  OptimizerPointer;

	void Execute(itk::Object *caller, const itk::EventObject & event) {
		Execute((const itk::Object *)caller, event);
	}

	void Execute(const itk::Object * object, const itk::EventObject & event) {
		OptimizerPointer optimizer =
			dynamic_cast< OptimizerPointer >(object);
		if (!itk::IterationEvent().CheckEvent(&event)) {
			return;
		}
		std::cout << "Iteration: " << optimizer->GetCurrentIteration() << "\t";
		std::cout << "Value: " << optimizer->GetValue() << std::endl;
		//std::cout << optimizer->GetCurrentPosition() << std::endl;
	}
};

int main(int argc, char *argv[])
{

	if (argc < 3) {
		cerr << "INPUT FAIL" << endl;
	}

	MetricType::Pointer metric = MetricType::New();
	TransformType::Pointer transform = TransformType::New();
	OptimizerType::Pointer optimizer = OptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	RegistrationType::Pointer registration = RegistrationType::New();

// READ FILES BEGIN
	ReaderType::Pointer fixed_reader = ReaderType::New();
	ReaderType::Pointer moving_reader = ReaderType::New();
	fixed_reader->SetFileName(argv[1]);
	moving_reader->SetFileName(argv[2]);
	fixed_reader->Update();
	moving_reader->Update();
	
	ImageType::Pointer fixed_image = fixed_reader->GetOutput();
	ImageType::Pointer moving_image = moving_reader->GetOutput();

	typedef itk::ImageDuplicator<ImageType> DuplicatorType;
	DuplicatorType::Pointer duplicator = DuplicatorType::New();

	duplicator->SetInputImage(moving_image);
	duplicator->Update();

	ImageType::Pointer original_moving_image = duplicator->GetOutput();
// READ FILES END


// NORMALIZE BEGIN
	typedef itk::NormalizeImageFilter<ImageType, ImageType> NormalizeFilterType;
	NormalizeFilterType::Pointer fixedNormalizer = NormalizeFilterType::New();
	NormalizeFilterType::Pointer movingNormalizer = NormalizeFilterType::New();

	fixedNormalizer->SetInput(fixed_image);
	movingNormalizer->SetInput(moving_image);

	fixedNormalizer->Update();
	movingNormalizer->Update();

	fixed_image = fixedNormalizer->GetOutput();
	moving_image = movingNormalizer->GetOutput();
// NORMALIZE END


// BLUR BEGIN
	typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussianFilterType;

	GaussianFilterType::Pointer fixedSmoother = GaussianFilterType::New();
	GaussianFilterType::Pointer movingSmoother = GaussianFilterType::New();

	fixedSmoother->SetVariance(2.0);
	movingSmoother->SetVariance(2.0);

	fixedSmoother->SetInput(fixed_image);
	movingSmoother->SetInput(moving_image);

	fixedSmoother->Update();
	movingSmoother->Update();

	fixed_image = fixedSmoother->GetOutput();
	moving_image = movingSmoother->GetOutput();
// BLUR END

	//cout << fixed_image->GetSpacing() << endl;
	//cout << fixed_image->GetBufferedRegion().GetNumberOfPixels() << endl;
	//cout << moving_image->GetBufferedRegion().GetNumberOfPixels() << endl << endl;

// OPTIMIZER PARAMETERS BEGIN
	optimizer->SetMaximumStepLength(0.1);
	optimizer->SetMinimumStepLength(0.0001);
	optimizer->SetNumberOfIterations(300);
	optimizer->MaximizeOn();
	//optimizer->MinimizeOn();

	double translationScale = 1.0 / 1000.0;
	typedef OptimizerType::ScalesType       OptimizerScalesType;
	OptimizerScalesType optimizerScales(transform->GetNumberOfParameters());

	optimizerScales[0] = 1.0;
	optimizerScales[1] = 1.0;
	optimizerScales[2] = 1.0;
	optimizerScales[3] = 1.0;
	optimizerScales[4] = 1.0;
	optimizerScales[5] = 1.0;
	optimizerScales[6] = 1.0;
	optimizerScales[7] = 1.0;
	optimizerScales[8] = 1.0;
	optimizerScales[9] = translationScale;
	optimizerScales[10] = translationScale;
	optimizerScales[11] = translationScale;
	optimizer->SetScales(optimizerScales);

	//typedef itk::CenteredTransformInitializer<TransformType, ImageType, ImageType > TransformInitializerType;
	//TransformInitializerType::Pointer initializer = TransformInitializerType::New();
	//initializer->SetTransform(transform);
	//initializer->SetFixedImage(fixed_image);
	//initializer->SetMovingImage(moving_image);
	//initializer->MomentsOn();
	//initializer->InitializeTransform();
	//registration->SetInitialTransformParameters(transform->GetParameters());

	ParametersType initialParameters(transform->GetNumberOfParameters());
	initialParameters[0] = 1.0;
	initialParameters[1] = 0.0;
	initialParameters[2] = 0.0;
	initialParameters[3] = 0.0;
	initialParameters[4] = 1.0;
	initialParameters[5] = 0.0;
	initialParameters[6] = 0.0;
	initialParameters[7] = 0.0;
	initialParameters[8] = 1.0;

	initialParameters[9] = 0.0;
	initialParameters[10] = 0.0;
	initialParameters[11] = 0.0;

	registration->SetInitialTransformParameters(initialParameters);

// OPTIMIZER PARAMETERS END

// METRIC PARAMETERS BEGIN
	metric->SetFixedImageStandardDeviation(0.4);
	metric->SetMovingImageStandardDeviation(0.4);

	const unsigned int numberOfPixels = fixed_image->GetBufferedRegion().GetNumberOfPixels();
	const unsigned int numberOfSamples = static_cast<unsigned int>(numberOfPixels * 0.0001);

	metric->SetNumberOfSpatialSamples(numberOfSamples);
// METRIC PARAMETERS END

// OBSERVER
	CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
	optimizer->AddObserver(itk::IterationEvent(), observer);
//

	//cout << "NSamples= " << numberOfSamples << endl;s

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetInterpolator(interpolator);
	registration->SetFixedImageRegion(fixed_image->GetBufferedRegion());

	
	

	//return 0;

	registration->SetFixedImage(fixed_image);
	registration->SetMovingImage(moving_image);

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
	resample->SetInput(original_moving_image);

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


