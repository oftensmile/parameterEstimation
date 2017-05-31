#include <cmath>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>	//inner_product
#include <valarray>	//slice
using namespace std;
vector<int> decimal_to_binary(int n);
vector<int> shift_vector(vector<int> v1);
void print_vec();

inline double rand01(){
	double r = (double)rand();
	return rand() / (RAND_MAX+1.0);
}

const int N = 8;                          // number of spins in x and y
vector<double> h0(N,1);     //	local magnetic field. 
vector<double> J0(N,1);       // spin-spin interaction.
vector<double> h(N,1);     // spin variable at each lattice site
vector<double> J(N,1);       //
vector<double> hist;
vector<int> spins(N,1);

vector<int> decimal_to_binary(int n){
	string binary = bitset<N>(n).to_string(); //to binary
	vector<int> v(N);
	for(int j=0; j<N;++j){
		v[j] = binary[j];
	}
	return v;
}

vector<int> shift_vector(vector<int> v1){
	vector<int> v2(v1);
	int head = *v2.begin();
	v2.erase(v2.begin());
	return v2; 
}

double E_tot_1d(){
	//vector<int> v=decimal_to_binary(n);
	double E=0;	
	for(int i=0;i<N;i++){
	E += -J[i]*spins[i]*spins[(i+1)%N]-h[i]*spins[i];	
	}	
	return E;
}

double dE_1d( int index){
	int dE;	
	dE = 2*(-spins[index])*(-0.5*J[index]*spins[(index+1)%N]
			-0.5*J[(index-1+N)%N]*spins[(index-1+N)%N]
			-h[index]*spins[index]);
	return dE;
}

double specific_heat(){
	vector<int> v1=shift_vector(spins);
	double temp = accumulate(spins.begin(),spins.end(),0);
	return inner_product(spins.begin(),spins.end(),v1.begin(),0)/N - temp*temp/N; 
}

// This function shuld be used for arbitary dimension.
void MP(int index,double beta){
	double dE, u, r; //dE = E_pop - E_prev
	int temp;
	dE = dE_1d(index);
	u = rand01();
	r=exp(-beta*dE);
	if(u<r){
	temp = spins[index];
       	spins[index]= -spins[index];
	}
}

void mc(double beta){
	int index;
	for(int i=0;i<N;i++){
		index = rand()%N;
		MP(index,beta);		
	}	
}
void print_vec(vector<int>){
	for(int i=0;i<N;++i){cout<<spins[i]<<" ";}
	cout<<endl;
}

int main(int argc, char* argv[]){
	int n_equribrium=2000,n_observ=200,n_auto=100;
	double beta,C=0,E=0,M=0;
	cout<<"Temperture:"<<endl;
	cin>>beta; beta = 1.0/beta;	
	cout<<"spins"<<endl;
	for(int t=0;t<n_equribrium;t++){
		mc(beta);	
	}
	for(int t=0;t<n_observ*n_auto;t++){
		mc(beta);	
		if(t%n_auto==0){
			C += specific_heat()/n_observ;
			E += E_tot_1d()/n_observ;
			M +=(double)accumulate(spins.begin(),spins.end(),0)/n_observ; 
		}	
	}
	cout<<"after update"<<endl;
	print_vec(spins);	
	cout<<"averaged magnetization: "<<M<<endl;
	cout<<"specific_heat: "<<C<<endl;
	cout<<"averaged energy:: "<<E<<endl;

}

