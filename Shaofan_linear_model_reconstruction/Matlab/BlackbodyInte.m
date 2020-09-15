function B = BlackbodyInte(T,L,f)
   lmin = min(L);
   lmax = max(L);
   lengthT = length(T);
   lengthL = length(L);
   sigma = (lmax - lmin)/(lengthL-1); % the factor is between FWHM and sigma

   % planck function
   c1 = 2*(6.63*1e-34)*(3*1e8)^2;   %2hC^2 the first radiation constant
   c2 = (6.63*1e-34)*(3*1e8)/(1.38*1e-23);  %hc/kB, the second radiation constant
   f_bb =@(t,l) (c1./(l.^5)./(exp(c2./(l*t))-1)/1e9);
   
   % calculate power desity at each wavelength - intergral of gaussian
   B = zeros(lengthT,lengthL);
   for i = 1:lengthT
       for j = 1:lengthL
           l = L(j);
           fun = @(x) (f_bb(T(i),x).*normpdf(x,L(j),sigma));
           B(i,j) = integral(fun,lmin-sigma,lmax+sigma);
       end
   end  

   
%  VPA
%    c1 = vpa(2*(6.63*1e-34)*(3*1e8)^2);   %2hC^2 the first radiation constant
%    c2 = vpa((6.63*1e-34)*(3*1e8)/(1.38*1e-23));  %hc/kB, the second radiation constant    
%    B = vpa(zeros(length(T),length(L)));

%  Filter information    
   if f == 1
       FilterSpectra = readtable('Filter.csv');
       Wavelgth = table2array(FilterSpectra(:,1))'/1e9;
       Trans = table2array(FilterSpectra(:,2))';
       InterTrans = interp1(Wavelgth,Trans,L,'linear','extrap')';
       for i = 1:length(T)
           B(i,:) = B(i,:).*InterTrans'/100;
       end
   end      
     
end