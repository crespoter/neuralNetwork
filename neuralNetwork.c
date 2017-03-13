/*
/_/                      

Neural network code to learn wine database .
/*
Headers
*/
#include<stdio.h>
#include<strings.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
	

//const double derivativeThreshold = 0.05;
const double stepSize = 0.9;//0.1;


/*function predefinitions */
double stof(const char*);
void assignRandomInner(double weights[][13],int length);
void assignRandoOuter(double weights[][3],int length);
void readFromFile(double featureX[][13],double testAnswers[]);
double stof(const char* s);
int forwardPropagate(double features[],double innerW[][13],double outerW[][3],int noOfHiddenElements,double inputForInner[],double outputFromInner[],double inputForOuter[],double answer[],double biasInput[],double biasOutput[]);
double updateInnerValues(double features[],double innerW[]);
void learn(double features[100][13],int sizeOfTrainingSet,double innerW[][13],double outerW[][3],int noOfHiddenElements,double inputForInner[],double outputFromInner[],double inputForOuter[],double biasInput[],double biasOutput[],double finalAnswer[][3],double trainAnswers[]);
double sigmund(double x);
double sigmundDerivative(double x);
void updateOuterLayer(double Error[],double outerW[][3],double outputFromInner[],double inputForInner[],double biasOutput[],int noOhHiddenLayers,double finalAnswers[]);

/*
Assign random values to weights array of length length
*/

void assignRandomBias(double weights[],int length)
{
	srand ( time(NULL) );
	int i;
	int j;
	for(i=0;i<length;i++)
	{
		
		weights[i] = 0*(rand()%9 + 1)/100.0;
		
	}
}
void assignRandomInner(double weights[][13],int length)
{
	srand ( time(NULL) );
	int i;
	int j;
	for(i=0;i<length;i++)
	{
		for(j=0;j<13;j++)
		{
			weights[i][j] = 0*(rand()%9 + 1)/100.0;
		}
	}
}
void assignRandomOuter(double weights[][3],int length)
{
	srand ( time(NULL) );
	int i;
	int j;
	for(i=0;i<length;i++)
	{
		for(j=0;j<3;j++)
		{
			weights[i][j] = 0*(rand()%9 + 1)/100.0;
		}
	}
}



/*
To convert string to double value . 
*/
double stof(const char* s){
  double rez = 0, fact = 1;
  if (*s == '-'){
    s++;
    fact = -1;
  };
  int point_seen;
  for (point_seen = 0; *s; s++){
    if (*s == '.'){
      point_seen = 1; 
      continue;
    };
    int d = *s - '0';
    if (d >= 0 && d <= 9){
      if (point_seen) fact /= 10.0f;
      rez = rez * 10.0f + (double)d;
    };
  };
  return rez * fact;
};

/*
Read data from the file train.csv and update featureX and testAnswers . 
featureX is the set of all features .
testAnswers in the set of the answers  .... ie Y 
*/

void readFromFile(double featureX[][13],double testAnswers[])
{
	FILE* fp;
  	char buf[2024];
	fp = fopen("train.csv", "r");
 	if(fp == NULL)
 	{
 		printf("ERROR");
	 }
	 int inputFeature = 0;
	 int inputNo = 0;
	while (fgets(buf, sizeof(buf), fp) != NULL)
	{
		inputFeature = 0;
	    buf[strlen(buf) - 1] = '\0';
	    char temp[20];
	    int i;
	    int j=0;
	    for(i=0;i<strlen(buf);i++)
		{
			if(buf[i]==',')
			{
				temp[j] = '\0';
				j=0;
				featureX[inputNo][inputFeature] = stof(temp);
				if(j==12)
				{
					featureX[inputNo][inputFeature] /= 1000;
				}
				if(j==0 || j==3)
				{
					featureX[inputNo][inputFeature] /= 10;
				}
				if(j==4)
				{
					featureX[inputNo][inputFeature] /= 100;
				}
				if(j==7)
				{
					featureX[inputNo][inputFeature] *= 10;
				}
				
					
				inputFeature++;
			}
			else
			{
				temp[j]=buf[i];
				j++;
			}
		}
		temp[j] = '\0';
		j=0;
		testAnswers[inputNo] = stof(temp);
		inputFeature++;
		
		
		inputNo++;
	  }
	  fclose(fp);
}


/*
This function updates the errors 
*/
void updateErrors(double finalAnswer[],double trainAnswers,double Error[])
{
	int i;
	
	if(trainAnswers <= 1)
	{
		Error[0] = 1-finalAnswer[0];
		Error[1] = 0-finalAnswer[0];
		Error[2] = 0-finalAnswer[0];
	}
	else if(trainAnswers <= 2)
	{
		Error[0] = 0-finalAnswer[0];
		Error[1] = 1-finalAnswer[0];
		Error[2] = 0-finalAnswer[0];
	}
	else if(trainAnswers <= 3)
	{
		Error[0] = 0-finalAnswer[0];
		Error[1] = 0-finalAnswer[0];
		Error[2] = 1-finalAnswer[0];
	}

}


/*
Update the weights
*/

void updateWeights(double Error[],double outerW[][3],double outputFromInner[],double inputForInner[],double biasOutput[],int noOfHiddenElements,double finalAnswers[] 
				,double innerW[][13],double biasInput[],double X[]) 
{
	
	int i,j;
	float temp = 0.0;
	double hiddenError[20];
	//COMPUTE ERROR FOR HIDDEN ELEMENTS
	for(i=0;i<noOfHiddenElements;i++)
	{
		for(j=0;j<3;j++)
		{
			temp += Error[j] * outerW[i][j];
		}
		hiddenError[i] = sigmundDerivative(outputFromInner[i]) * temp;
		temp = 0.0;
	}
	double length = 0.0;
	for(i=0;i<noOfHiddenElements;i++)
	{
		length += outputFromInner[i] * outputFromInner[i];
	}
	if(length <= 0.1)
	{
		length = 0.1;
	}
	for(i=0;i<noOfHiddenElements;i++)
	{
		for(j=0;j<3;j++)
		{
			outerW[i][j] += ( stepSize * Error[j] * outputFromInner[i] / length);
		}
	}
	//Adjust bias of output
	for(i=0;i<3;i++)
	{
		biasOutput[i] += (stepSize * Error[i]/length); 
	}
	
	//FROM INPUT TO HIDDEN
	
	length = 0.0;
	for(i=0;i<13;i++)
	{
		length += X[i]*X[i];	
	}
	if(length <=0.1)
	{
		length = 0.1;
	}
	for(i=0;i<13;i++)
	{
		for(j=0;j<noOfHiddenElements;j++)
		{
			innerW[j][i] += (stepSize * hiddenError[j] * X[i] /length);
		}
	}
	//ADJUST BIAS FOR HIDDEN
	for(i=0;i<noOfHiddenElements;i++)
	{
		biasInput[i] += (stepSize * hiddenError[i]/length); 
	}
	
}

	


/*
Calling function is 	learn(featureX,sizeOfTrainingSet,innerW,outerW,noOfHiddenElements,inputForInner,outputFromInner,inputForOuter,biasInput,biasOutput,finalAnswer,trainAnswers);
The learning function
calls other functions necessary for learning 
*/


void learn(double features[100][13],int sizeOfTrainingSet,double innerW[][13],double outerW[][3],int noOfHiddenElements,double inputForInner[],double outputFromInner[],double inputForOuter[],double biasInput[],double biasOutput[],double finalAnswer[][3],double trainAnswers[])
{
	
	int i;
	double Error[3];
	int j=0;
	do
	{
		
		for(i=0;i<sizeOfTrainingSet;i++)
		{
		
			forwardPropagate(features[i],innerW,outerW,noOfHiddenElements,inputForInner,outputFromInner,inputForOuter,finalAnswer[i],biasInput,biasOutput);	
			updateErrors(finalAnswer[i],trainAnswers[i],Error);
			updateWeights(Error,outerW,outputFromInner,inputForOuter,biasOutput,noOfHiddenElements,finalAnswer[i],innerW,biasInput,features[i]);
		}
		j++;
	}
	while(j<60000);


	for(i=0;i<118;i++)
	{
		printf("%d %f\n",forwardPropagate(features[0],innerW,outerW,noOfHiddenElements,inputForInner,outputFromInner,inputForOuter,finalAnswer[0],biasInput,biasOutput),trainAnswers[i]);
	}

}



/*
Propagates the function forward updating the inputforinner,outputfrominner,inputforouter and answer .
*/
int forwardPropagate(double features[],double innerW[][13],double outerW[][3],int noOfHiddenElements,double inputForInner[],double outputFromInner[],double inputForOuter[],double answer[],double biasInput[],double biasOutput[])
{
	int i;
	//Hidden layer calculations.
	for(i=0;i<noOfHiddenElements;i++)
	{
		inputForInner[i] = updateInnerValues(features,innerW[i]) + biasInput[i];
		
		outputFromInner[i]= sigmund(inputForInner[i]);
	}
	
	//Outer layer calculations
	for(i=0;i<3;i++)
	{
		int j;
		double ans = 0;
		for(j=0;j<noOfHiddenElements;j++)
		{
			ans += (outputFromInner[j]*outerW[j][i]);
		}
		inputForOuter[i] = ans + biasOutput[i];
		answer[i] = sigmund(ans);
	}
	if(answer[0]>answer[1])
	{
		return 1;
	}
	else if(answer[1]>answer[2])
	{
		return 2;
	}
	else
	{
		return 3;
	}
}


double updateInnerValues(double features[],double innerW[])
{
	int i;
	double answer = 0;
	for(i=0;i<13;i++)
	{
		answer += (features[i]*innerW[i]);
	}
	return answer;
}


/*
Sigmund and its derivative function definitions
*/
double sigmund(double x)
{
	return 1/(1+pow(2.718,-x));
}
double sigmundDerivative(double sigmundValue)
{
	return sigmundValue * (1-sigmundValue);
}



int main()
{
	const int noOfHiddenElements = 20;
	const int sizeOfTrainingSet = 118;
	double featureX[500][13];
	double trainAnswers[500];
	double innerW[20][13];
	double outerW[20][3];
	double inputFeatures[13];
	double inputForInner[20];
	double outputFromInner[20];
	double inputForOuter[20];
	double finalAnswer[20][3];
	double biasInput[20];
	double biasOutput[3];

	
	/* 
	read the features and its corresponding class from train.csv file 
	*/
	readFromFile(featureX,trainAnswers);  
	/*
	assign random weights 
	*/
	
	assignRandomInner(innerW,noOfHiddenElements);
	assignRandomOuter(outerW,noOfHiddenElements);
	assignRandomBias(biasInput,noOfHiddenElements);
	assignRandomBias(biasOutput,3);
	/*
	calling the learning function
	*/
	learn(featureX,sizeOfTrainingSet,innerW,outerW,noOfHiddenElements,inputForInner,outputFromInner,inputForOuter,biasInput,biasOutput,finalAnswer,trainAnswers);
	return 0;
}
