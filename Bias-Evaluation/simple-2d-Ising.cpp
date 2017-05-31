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

struct parameter{
	double x;
	double y;
};

inline double rand01(){
	double r = (double)rand();
	return rand() / (RAND_MAX+1.0);
}

const int Nx = 8;                          // number of spins in x 
const int Ny = 8;                          // number of spins in y
const int N = Nx * Ny;

vector< vector<double> > h(Nx, vector<double>(Ny,1));
vector< vector<parameter> > J(Nx, vector<parameter>(Ny));
vector< vector<double> > h0(Nx, vector<double>(Ny,1));
vector< vector<parameter> > J0(Nx, vector<parameter>(Ny));

vector<double> hist(N,0);
vector< vector<int> > spins(Ny, vector<int>(Nx,1));

void init(){
	for(int i=0;i<Nx;++i){
	for(int j=0;j<Ny;++j){
	J[i][j].x= 1.0;	
	J[i][j].y= 1.0;	
	J0[i][j].x= 1.0;	
	J0[i][j].y= 1.0;	
	}	
	}
}
/*
vector<int> decimal_to_binary(int n){
	string binary = bitset<N>(n).to_string(); //to binary
	vector<int> v(N);
	for(int j=0; j<N;++j){
		v[j] = binary[j];
	}
	return v;
}
*/
/*
vector<int> shift_vector(vector<int> v1){
	vector<int> v2(v1);
	int head = *v2.begin();
	v2.erase(v2.begin());
	return v2; 
}
*/

double E_tot_2d(){
	//vector<int> v=decimal_to_binary(n);
	double E=0;	
	for(int i=0;i<Nx;i++){
	for(int j=0;j<Ny;j++){
	E += -spins[i][j]*(J[i][j].x * spins[(i+1)%Nx][j]
			  +J[i][j].y * spins[i][(j+1)%Ny]
			+h[i][j]
			);
	}	
	}	
	return E;
}

double dE_2d(int index_x,int index_y){
	int dE;	
	dE = 2*(-spins[index_x][index_y])*(
		-0.5*J[index_x][index_y].x * spins[(index_x+1)%Nx][index_y]
		-0.5*J[(index_x-1+Nx)%Nx][index_y].x * spins[(index_x-1+Nx)%Nx][index_y]
		
		-0.5*J[index_x][index_y].y * spins[index_x][(index_y+1)%Ny]
		-0.5*J[index_x][(index_y-1+Ny)%Ny].y * spins[index_x][(index_y-1+Ny)%Ny]
		
		-h[index_x][index_y]);
	return dE;
}
/*
double specific_heat_2d(){
	vector<int> v1=shift_vector(spins);
	double temp = accumulate(spins.begin(),spins.end(),0);
	return inner_product(spins.begin(),spins.end(),v1.begin(),0)/N - temp*temp/N; 
}
*/
// This function shuld be used for arbitary dimension.
void MP_2d(int index_x,int index_y,double beta){
	double dE, u, r; //dE = E_pop - E_prev
	int temp;
	dE = dE_2d(index_x,index_y);
	u = rand01();
	r=exp(-beta*dE);
	if(u<r){
       	spins[index_x][index_y]= -spins[index_x][index_y];
	}
}

void mc(double beta){
	int index_x,index_y;
	for(int i=0;i<Nx;i++){
		for(int j=0;j<Ny;j++){
		index_x = rand()%Nx;
		index_y = rand()%Ny;
		MP_2d(index_x,index_y,beta);		
	}	
	}	
}
void print_vec_2d(vector< vector<int> > v){
	for(int i=0;i<Nx;++i){
	for(int j=0;j<Ny;++j){cout<<spins[i][j]<<" ";}
	cout<<endl;
	}
}

double calc_M(){
	double M = 0;
	for(int i=0;i<Nx;++i){
	M+=(double)accumulate(spins[i].begin(),spins[i].end(),0);
	}
	return M;
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
			//C += specific_heat()/n_observ;
			E += E_tot_2d()/n_observ;
			M +=calc_M()/n_observ; 
		}	
	}
	cout<<"after update"<<endl;
	print_vec_2d(spins);	
	cout<<"averaged magnetization: "<<M<<endl;
	//cout<<"specific_heat: "<<C<<endl;
	cout<<"averaged energy:: "<<E<<endl;

}

