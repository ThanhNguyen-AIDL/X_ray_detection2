import React from 'react';

interface TMALogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  className?: string;
}

export const TMALogo: React.FC<TMALogoProps> = ({ 
  size = 'md', 
  showText = true, 
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'h-8',
    md: 'h-12', 
    lg: 'h-16'
  };

  const textSizeClasses = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-4xl'
  };

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* TMA Logo Image */}
      <img 
        src="/logo-menu.webp" 
        alt="TMA Logo" 
        className={`${sizeClasses[size]} object-contain`}
      />
      
      {/* Company Text */}
      {showText && (
        <div className="flex flex-col">
          <h1 className={`font-bold ${textSizeClasses[size]} text-foreground`}>
            TMA Solutions
          </h1>
          <p className="text-xs text-muted-foreground font-medium">
            Tập đoàn Công nghệ hàng đầu Việt Nam
          </p>
        </div>
      )}
    </div>
  );
};