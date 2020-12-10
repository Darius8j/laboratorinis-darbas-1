%=========================================1 užduotys==========================================
% Sukurkite paprasto klasifikatoriaus (Perceptrono) išėjimui apskaičiuoti skirtą programą. 
% Klasifikatorius turi skirstyti objektus į dvi klases, pagal du požymius. 
% Išėjimo skaičiavimas atliekamas pagal formulę: 
% y = 1, kai x1*w1 + x2*w2 + b > 0; 
% y = -1, kai x1*w1 + x2*w2 + b <= 0; 
% čia w1, w2 ir b parametrai, kurie turi būti sugeneruojami naudojant atsitiktinių skaičių generatorių
% (MATLAB pvz.: w1 = randn(1);) pirmąją programos veikimo iteraciją ir vėliau atnaujinami mokymo algoritmu; 
% x1 ir x2 yra objektų požymiai, apskaičiuoti Matlab funkcijomis, 
% esančiomis paruoštame kodo ruošinyje arba Data.txt faile 
% (kiekvienoje eilutėje yra toks duomenų formatas: požymis1, požymis2, norimas_atsakymas), 
% jei ketinate naudoti ne Matlab.
%=========================================2 užduotys=============================================
% Parašykite mokymo algoritmą sukurto klasifikatoriaus parametrams apskaičiuoti. 
% Naudokite šią parametrų atnaujinimo formulę: w1(n+1) = w1(n) + eta*e(n)*x1(n); 
% čia 0 < eta < 1; e(n) = d(n) - y(n); - klaida (momentinė klaida), 
% apskaičiuota palyginus norimą atsakymą su tuo, kuris gautas klasifikatoriaus išėjime. 
% w2(n+1) = w2(n) + eta*e(n)*x2(n); b(n+1) = b(n) + eta*e(n);
%-------------------------------------------------------------------------------------------
%-------------------------------------------------------------------------------------------
% ====Klasifikatorius turi skirstyti objektus į dvi klases, pagal du požymius.
% ====Pozymiai (obuoliai ir kriauses)
%obuoliai
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%kriauses
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');
%-------------------------------------------------------------------------------------------

%Calculate for each image, colour and roundness
%For Apples---------------------------------------------------------------------------------
%1st apple image(A1)
hsv_value_A1=A1; %color
metric_A1=A1; %roundness
%2nd apple image(A2)
hsv_value_A2=A2; %color
metric_A2=A2; %roundness
%3rd apple image(A3)
hsv_value_A3=A3; %color
metric_A3=A3; %roundness
%4th apple image(A4)
hsv_value_A4=A4; %color
metric_A4=A4; %roundness
%5th apple image(A5)
hsv_value_A5=A5; %color
metric_A5=A5; %roundness
%6th apple image(A6)
hsv_value_A6=A6; %color
metric_A6=A6; %roundness
%7th apple image(A7)
hsv_value_A7=A7; %color
metric_A7=A7; %roundness
%8th apple image(A8)
hsv_value_A8=A8; %color
metric_A8=A8; %roundness
%9th apple image(A9)
hsv_value_A9=A9; %color
metric_A9=A9; %roundness
%For Pears---------------------------------------------------------------------------
%1st pear image(P1)
hsv_value_P1=P1; %color
metric_P1=P1; %roundness
%2nd pear image(P2)
hsv_value_P2=P2; %color
metric_P2=P2; %roundness
%3rd pear image(P3)
hsv_value_P3=P3; %color
metric_P3=P3; %roundness
%4nd pear image(P4)
hsv_value_P4=P4; %color
metric_P4=P4; %roundness
%selecting features(color, roundness, 3 apples and 2 pears)

%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];
%Desired output vector
T=[1;1;1;-1;-1];

%train single perceptron with two inputs and one output
% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

nu = 0.25;
W = [w1 w2];

% Išėjimo skaičiavimas atliekamas pagal formulę: 
% y = 1, kai x1*w1 + x2*w2 + b > 0; 
% y = -1, kai x1*w1 + x2*w2 + b <= 0; 
% calculate wieghted sum with randomly generated parameters
%------------------------------
v1 = x1(1)*w1+x2(1)*w2+b;

if v1 > 0
	y = 1;
else
	y = -1; 
end
e1 = T(1) - y;

W(1) = W(1)+nu*e1*x1(1);
W(2) = W(2)+nu*e1*x2(1);
b = b+nu*e1;
%------------------------------
v2 = x1(2)*w1+x2(2)*w2+b; 
if v2 > 0
	y = 1;
else
	y = -1;
end
e2 = T(2) - y;

W(1) = W(1)+nu*e2*x1(2);
W(2) = W(2)+nu*e2*x2(2);
b = b+nu*e2;
%------------------------------
v3 = x1(3)*w1+x2(3)*w2+b;
if v3 > 0
	y = 1;
else
	y = -1; 
end
e3 = T(3) - y;

W(1) = W(1)+nu*e3*x1(3);
W(2) = W(2)+nu*e3*x2(3);
b = b+nu*e3;
%------------------------------
v4 = x1(4)*w1+x2(4)*w2+b; 
if v4 > 0
	y = 1;
else
	y = -1; 
end
e4 = T(4) - y;

W(1) = W(1)+nu*e4*x1(4);
W(2) = W(2)+nu*e4*x2(4);
b = b+nu*e4;
%------------------------------
v5 = x1(5)*w1+x2(5)*w2+b;
if v5 > 0
	y = 1;
else
	y = -1; 
end

e5 = T(5) - y;

W(1) = W(1)+nu*e5*x1(5);
W(2) = W(2)+nu*e5*x2(5);
b = b+nu*e5;
%------------------------------




% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);


while e ~= 0
for i=5
  
v = x1(i)*W(1)+x2(i)*W(2)+b;
if v > 0
    y = 1;
else 
    y = -1;
end
 e = T(i) - y;

    W(1) = W(1)+nu*e*x1(i);
    W(2) = W(2)+nu*e*x2(i);
    b = b+nu*e;
end
end

%----patikrinimas-----------------
for z = 1:5
   v = x1(z)*W(1)+x2(z)*W(2)+b;
if v > 0
    y = 1;
else 
    y = -1;
    
end
 e11 = T(z) - y; 
    
end
