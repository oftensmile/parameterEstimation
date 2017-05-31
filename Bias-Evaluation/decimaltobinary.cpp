#include <iostream>
#include <vector>
#include <bitset>

int main()
{
	int num;
	std::cout<<"input decimal"<<std::endl;
	std::cin>>num;
	std::string binary = std::bitset<8>(num).to_string(); //to binary
	std::cout<<"binary="<<binary<<"\n";

	unsigned long decimal = std::bitset<8>(binary).to_ulong();
	std::cout<<decimal<<"\n";
	
	std::cout<<"biary[0]="<<binary[0]<<"\n";
	std::cout<<"biary[7]="<<binary[7]<<"\n";
	std::vector<int> v(8);
	for(int j=0; j<binary.size();++j){
		v[j] = binary[j];
	}
	
	for(int j=0; j<v.size();++j){
		std::cout<<"vector["<<j<<"]="<<binary[j]<<"\n";
	}	
	return 0;
}
