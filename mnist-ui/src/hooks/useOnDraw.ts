import { useEffect, useRef } from "react";
import { PIXEL_SIZE } from "../App";

export type Point = {
        x: number;
        y: number;
    } | null;

export function useOnDraw(onDraw:any) {

    const canvasRef = useRef<HTMLCanvasElement|null>(null);
    const isDrawingRef = useRef(false);
    const prevPointRef = useRef<Point>(null);

    const mouseMoveListenerRef = useRef<((e: MouseEventInit) => void)|null>(null);
    const mouseUpListenerRef = useRef<((e: MouseEventInit) => void)|null>(null);

    function setCanvasRef(ref:HTMLCanvasElement) {
        canvasRef.current = ref;
    }

    function onCanvasMouseDown() {
        isDrawingRef.current = true;
    }

    function clearCanvas(){
        if(canvasRef.current){
            const ctx = canvasRef.current.getContext('2d');
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        }
    }

    function getImage(){
        if(canvasRef.current){
            const canvasWidth = canvasRef.current.width;
            const canvasHeight = canvasRef.current.height
            const ctx = canvasRef.current.getContext('2d');
            const image = ctx?.getImageData(
                0, 0, canvasWidth,canvasHeight,{colorSpace:"srgb"})as ImageData
            let flattenImage = []
            let str = ""
            for (let row = 0; row < 28; row++) {
                for (let col = 0; col < 28; col++) {
                    const pixel = calcOnePixel(image,row,col,canvasWidth);
                    flattenImage.push(pixel/255)
                    if(pixel>50)str+='# '
                    else str+='. '
                } 
                str+='\n'
            }
            console.log(str);
            
            return flattenImage;
        }
    }
    function calcOnePixel(image:ImageData,row:number,col:number,canvasWidth:number){
        let avarageValue = 0;
        const rowBound = 4*(row+1)*PIXEL_SIZE;
        const colBound = 4*(col+1)*PIXEL_SIZE;
        for (let rowInd = 4*row*PIXEL_SIZE; rowInd <rowBound; rowInd+=4) {
            for (let colInd = 4*col*PIXEL_SIZE; colInd <colBound; colInd+=4) {
                const ind = canvasWidth*rowInd+colInd;
                const red = image.data[ind];
                const green = image.data[ind];
                const blue = image.data[ind];
                avarageValue+= (red+green+blue)/3
            }
        }
        return avarageValue/(PIXEL_SIZE*PIXEL_SIZE)
    }

    useEffect(() => {
        function computePointInCanvas(clientX:number, clientY:number) {
            if (canvasRef.current) {
                const boundingRect = canvasRef.current.getBoundingClientRect();
                return {
                    x: clientX - boundingRect.left,
                    y: clientY - boundingRect.top
                }
            } else {
                return null;
            }

        }
        function initMouseMoveListener() {
            const mouseMoveListener = (e:(MouseEventInit)) => {
                if (isDrawingRef.current && canvasRef.current) {
                    const point = computePointInCanvas(e.clientX as number, e.clientY  as number);
                    const ctx = canvasRef.current.getContext('2d');
                    if (onDraw) onDraw(ctx, point, prevPointRef.current);
                    prevPointRef.current = point;
                }
            }
            mouseMoveListenerRef.current = mouseMoveListener;
            window.addEventListener("mousemove", mouseMoveListener);
        }

        function initMouseUpListener() {
            const listener = () => {
                isDrawingRef.current = false;
                prevPointRef.current = null;
            }
            mouseUpListenerRef.current = listener;
            window.addEventListener("mouseup", listener);
        }

        function cleanup() {
            if (mouseMoveListenerRef.current) {
                window.removeEventListener("mousemove", mouseMoveListenerRef.current);
            }
            if (mouseUpListenerRef.current) {
                window.removeEventListener("mouseup", mouseUpListenerRef.current);
            }
        }

        initMouseMoveListener();
        initMouseUpListener();
        return () => cleanup();

    }, [onDraw]);

    return {
        setCanvasRef,
        onCanvasMouseDown,
        clearCanvas,
        getImage
    }

};