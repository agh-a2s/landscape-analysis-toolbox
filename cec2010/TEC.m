function [f,g,h] = TEC(x,I_fno)

 [ps,D]=size(x);

 global initial_flag
 persistent o M

 if(I_fno == 1)

      if initial_flag == 0
       load Function1
       o = o(1:D);
      end
  x = x - repmat(o,size(x,1),1);
  f1 = (sum(cos(x).^4,2) - 2* prod(cos(x).^2,2));
  f2 = zeros(ps,1);
  for i=1:D
      f2=f2+i*x(:,i).^2;
  end
  f = -abs(f1./sqrt(f2));
  g1 = 0.75-prod(x,2);
  g2 = sum(x,2)-7.5*D;
  g = [g1,g2];
  h =zeros(ps,1);
 end


if(I_fno==2)

      if initial_flag == 0
       load Function2
       o = o(1:D);
      end
   x = x - repmat(o,size(x,1),1);
   f = max(x,[],2);
   y = x-0.5;
   g1= 10-sum(x.^2-10.*cos(2.*pi.*x)+10,2)/D;
   g2= sum(x.^2-10.*cos(2.*pi.*x)+10,2)/D-15;
   g = [g1,g2];
   h = sum(y.^2-10.*cos(2.*pi.*y)+10,2)/D-20;
end

if(I_fno == 3)
      if initial_flag == 0
       load Function3
       o = o(1:D);
      end
   x=x-repmat(o,size(x,1),1);
   f = sum(100*(x(:,1:D-1).^2-x(:,2:D)).^2+(x(:,1:D-1)-1).^2,2);
   g = zeros(ps,1);
   h = sum((x(:,1:D-1)-x(:,2:D)).^2,2);
end

if(I_fno==4)
      if initial_flag == 0
       load Function4
       o = o(1:D);
      end
   x = x - repmat(o,size(x,1),1);
   f = max(x,[],2);
   g = zeros(ps,1);
   h1 = sum(x.*cos(sqrt(abs(x))),2)/D;
   h2 = sum((x(:,1:D/2-1)-x(:,2:D/2)).^2,2);
   h3 = sum((x(:,D/2+1:D-1).^2-x(:,D/2+2:D)).^2,2);
   h4 = sum(x,2);
   h = [h1,h2,h3,h4];
end

if(I_fno==5)

     if initial_flag == 0
       load Function5
       o = o(1:D);
      end

  x = x-repmat(o,size(x,1),1);
  f = max(x,[],2);
  g = zeros(ps,1);
  h1 = sum(-x.*sin(sqrt(abs(x))),2)/D;
  h2 = sum(-x.*cos(0.5*sqrt(abs(x))),2)/D;
  h = [h1,h2];
end

if(I_fno==6)

    if initial_flag == 0
       load Function6
       o = o(1:D);
       if(D==10)
           load Function6_M_10D
       end
       if(D==30)
           load Function6_M_30D
       end
    end

  z = x - repmat(o,size(x,1),1);
  f = max(z,[],2);
  y = (x + 483.6106156535 - repmat(o,size(x,1),1));
  %y = y*M;
  y = y - 483.6106156535;
  g = zeros(ps,1);
  h1 = sum(-y.*sin(sqrt(abs(y))),2)/D;
  h2 = sum(-y.*cos(0.5*sqrt(abs(y))),2)/D;
  h = [h1,h2];
end

if(I_fno==7)

     if initial_flag == 0
       load Function7
       o = o(1:D);
      end
  x = x - repmat(o,size(x,1),1) ;
  z = x + 1;
  f = sum(100*(z(:,1:D-1).^2-z(:,2:D)).^2+(z(:,1:D-1)-1).^2,2);
  g = 0.5-exp(-0.1.*sqrt(sum(x.^2,2)./D))-3*exp(sum(cos(0.1*x),2)./D)+exp(1);
  h = zeros(ps,1);

end

if(I_fno==8)
    if initial_flag == 0
       load Function8
       o = o(1:D);
       if(D==10)
           load Function8_M_10D
       end
       if(D==30)
           load Function8_M_30D
       end
    end

  z = x + 1 - repmat(o,size(x,1),1);
  f = sum(100*(z(:,1:D-1).^2-z(:,2:D)).^2+(z(:,1:D-1)-1).^2,2);
  y = (x - repmat(o,size(x,1),1))*M;
  g = 0.5-exp(-0.1.*sqrt(sum(y.^2,2)./D))-3*exp(sum(cos(0.1*y),2)./D)+exp(1);
  h = zeros(ps,1);

 end

if(I_fno ==9)
      if initial_flag == 0
       load Function9
       o = o(1:D);
      end
  x = x - repmat(o,size(x,1),1);
  z = x + 1;
  f  = sum(100*(z(:,1:D-1).^2-z(:,2:D)).^2+(z(:,1:D-1)-1).^2,2);
  h = sum(x.*sin(sqrt(abs(x))),2);
  g  = zeros(ps,1);

end

if(I_fno ==10)

   if initial_flag == 0
       load Function10
       o = o(1:D);
       if(D==10)
           load Function10_M_10D
       end
       if(D==30)
           load Function10_M_30D
       end
   end
  z = x + 1 - repmat(o,size(x,1),1);
  f = sum(100*(z(:,1:D-1).^2-z(:,2:D)).^2+(z(:,1:D-1)-1).^2,2);
  y = (x - repmat(o,size(x,1),1))*M;
  h = sum(y.*sin(sqrt(abs(y))),2);
  g = zeros(ps,1);

end



if(I_fno ==11)

    if initial_flag == 0
       load Function11
       o = o(1:D);
       if(D==10)
           load Function11_M_10D
       end
       if(D==30)
           load Function11_M_30D
       end
    end
   z = (x - repmat(o,size(x,1),1))*M;
   y = x + 1 - repmat(o,size(x,1),1);
   f = sum(-z.*cos(2*sqrt(abs(z))),2)/D;
   h = sum(100*(y(:,1:D-1).^2-y(:,2:D)).^2+(y(:,1:D-1)-1).^2,2);
   g = zeros(ps,1);

end


if(I_fno == 12)

    if initial_flag == 0
       load Function12
       o = o(1:D);
    end

  f = sum(x.*sin(sqrt(abs(x))),2);
  h = sum((x(:,1:D-1).^2-x(:,2:D)).^2,2);
  g = sum(x-100.*cos(0.1.*x)+10,2);
  end


if(I_fno==13)
   if initial_flag == 0
       load Function13
       o = o(1:D);
    end
  x=x-repmat(o,size(x,1),1);
  f=sum(-x.*sin(sqrt(abs(x))),2)/D;
  g3=ones(ps,1);
  for i=1:D
     g3=g3.*cos(x(:,i)./sqrt(i));
  end
  g3=(75-50*(sum(x.^2,2)./4000-g3+1));
  g = [-50+sum(x.^2,2)/(D*100),50*sum(sin(1/50*pi*x),2)/D,g3];
  h = zeros(ps,1);
end

 if(I_fno==14)

     if initial_flag == 0
       load Function14
       o = o(1:D);
     end

  z = x + 1;
  f = sum(100*(z(:,1:D-1).^2-z(:,2:D)).^2+(z(:,1:D-1)-1).^2,2);
  g1= sum(-x.*cos(sqrt(abs(x))),2)-D;
  g2= sum(x.*cos(sqrt(abs(x))),2)-D;
  g3= sum(x.*sin(sqrt(abs(x))),2)-10*D;
  g = [g1,g2,g3];
  h =zeros(ps,1);
 end


 if(I_fno==15)

     if initial_flag == 0
       load Function15
       o = o(1:D);
       if(D==10)
           load Function15_M_10D
       end
       if(D==30)
           load Function15_M_30D
       end
    end

  z = x + 1;
  f = sum(100*(z(:,1:D-1).^2-z(:,2:D)).^2+(z(:,1:D-1)-1).^2,2);
  x = x*M;
  g1= sum(-x.*cos(sqrt(abs(x))),2)-D;
  g2= sum(x.*cos(sqrt(abs(x))),2)-D;
  g3= sum(x.*sin(sqrt(abs(x))),2)-10*D;
  g = [g1,g2,g3];
  h =zeros(ps,1);
 end





 if(I_fno==16)
     if initial_flag == 0
       load Function16
       o = o(1:D);
     end
  x =x - repmat(o,size(x,1),1);
  f=ones(ps,1);
  for i=1:D
     f=f.*cos(x(:,i)./sqrt(i));
  end
  f=(sum(x.^2,2)./4000-f+1);
  h1= sum(-x.*sin(sqrt(abs(x))),2);
  h2= sum(x.*sin(sqrt(abs(x))),2);
  g1 = sum(x.^2-100.*cos(pi.*x)+10,2);
  g2 = prod(x,2);
  g = [g1,g2];
  h = [h1,h2];

 end

 if(I_fno==17)

    if initial_flag == 0
       load Function17
       o = o(1:D);
    end
  x = x - repmat(o,size(x,1),1);
  f = sum((x(:,1:D-1)-x(:,2:D)).^2,2);
  g1 = prod(x,2);
  g2 = sum(x,2);
  g = [g1,g2];
  h = sum(x.*sin(4*sqrt(abs(x))),2);

 end

  if(I_fno==18)

     if initial_flag == 0
       load Function18
       o = o(1:D);
     end

  f = sum((x(:,1:D-1)-x(:,2:D)).^2,2);
  g= sum(-x.*sin(sqrt(abs(x))),2)/D;
  h= sum(x.*sin(sqrt(abs(x))),2)/D;

 end
