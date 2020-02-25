function neighborhood(vector1,vector2,r,c)
%% Function to connect with ray all neighborhood coordinates

    hold off;
    for i=1:(r*c)
        neighbor(i,vector1,vector2,r,c);
    end
    
    for p=1:(r*c)
        plot(vector1(p),vector2(p),'r*');
    end
end
